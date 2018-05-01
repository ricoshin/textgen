import logging
from enum import Enum, auto, unique

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from loader.data import Batch
from models.base_module import BaseModule
from nn.bnlstm import LSTM, BNLSTMCell
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from utils.utils import to_gpu
from utils.writer import ResultWriter

log = logging.getLogger('main')


class BaseDecoder(BaseModule):
    def __init__(self):
        super(BaseDecoder, self).__init__()

    def forward(self, code, batch=None, max_len=None):
        return self._decode(DecoderInPack(code, batch, max_len))

    def clip_grad_norm(self):
        nn.utils.clip_grad_norm(self.parameters(), self.cfg.clip)
        return self

    def make_noise_size_of(self, *size):
        noise = Variable(torch.ones(*size))
        noise = to_gpu(self.cfg.cuda, noise)
        noise.data.normal_(0, 1)
        return noise


class DecoderInPack(object):
    def __init__(self, code, batch=None, max_len=None):
        if batch is not None:
            assert isinstance(batch, Batch)

        if batch is not None and max_len is not None:
            raise Exception("batch and max_len has to be set alternatively")

        if batch is not None:
            self._rnn_mode = DecoderRNN.Mode.TEACHER_FORCE
        elif max_len is not None:
            self._rnn_mode = DecoderRNN.Mode.FREE_RUN
        else:
            self._rnn_mode = None

        self._code = code
        self._batch = batch
        self._max_len = max_len

    @property
    def code(self):
        if self._code is None:
            raise Exception('DecoderInPack.code is unset')
        return self._code

    @property
    def batch(self):
        if self._batch is None:
            raise Exception('DecoderInPack.batch is unset')
        return self._batch

    @property
    def max_len(self):
        if self._max_len is None:
            raise Exception('DecoderInPack.max_len is unset')
        return self._max_len

    @property
    def rnn_mode(self):
        if self._rnn_mode is None:
            raise Exception("DecoderInPack.rnn_mode is unset")
        return self._rnn_mode


class DecoderRNN(BaseDecoder):
    @unique
    class Mode(Enum):
        TEACHER_FORCE = auto()
        FREE_RUN = auto()

    def __init__(self, cfg, embed_w):
        super(DecoderRNN, self).__init__()
        self.cfg = cfg
        self.embed_w = embed_w
        self.vocab_w = embed_w.vocab
        self.packer_w = DecoderOutPackerRNN(cfg, embed_w.vocab)

        input_size = cfg.embed_size_w + cfg.hidden_size_w
        self.decoder = nn.LSTM(input_size=input_size,
                               hidden_size=cfg.hidden_size_w,
                               num_layers=1,
                               dropout=cfg.dropout,
                               batch_first=True)
        if cfg.dec_embed:
            self.linear_w = nn.Linear(cfg.hidden_size_w, cfg.embed_size_w)
        else:
            self.linear_w = nn.Linear(cfg.hidden_size_w, cfg.vocab_size_w)

        self.criterion_ce = nn.CrossEntropyLoss()
        self.criterion_nll = nn.NLLLoss()
        self._init_weights()

    def _decode(self, inpack):
        assert isinstance(inpack, DecoderInPack)
        if inpack.rnn_mode == self.Mode.TEACHER_FORCE:
            decoded = self._decode_teacher_force(
                inpack.code, inpack.batch.src, inpack.batch.len)
        elif inpack.rnn_mode == self.Mode.FREE_RUN:
            decoded = self._decode_free_run(inpack.code, inpack.max_len)
        return decoded

    def _decode_teacher_force(self, code_w, words, lengths):
        batch_size = code_w.size(0)

        # insert sos in the first column
        sos_w = self._get_sos_batch(batch_size, self.vocab_w)
        words = torch.cat([sos_w, words], 1)

        # length should be increased as well
        lengths = [length + 1 for length in lengths]
        init_state_w = self._init_hidden(batch_size, self.cfg.hidden_size_w)

        embed_in_w = self.embed_w(words)  # for teacher forcing
        all_code_w = code_w.unsqueeze(1).repeat(1, max(lengths), 1)

        # Decoder
        input_w = torch.cat([embed_in_w, all_code_w], 2)
        packed_input_w = pack_padded_sequence(input=input_w,
                                              lengths=lengths,
                                              batch_first=True)
        packed_output_w, _ = self.decoder(packed_input_w, init_state_w)
        output_w, length_w = pad_packed_sequence(
            packed_output_w, batch_first=True)

        # output layer for word prediction
        if self.cfg.dec_embed:
            embed_out_w = self.linear_w(output_w)
            cosim_w = self._compute_cosine_sim(embed_out_w, self.embed_w.embed)
            prob_w = F.log_softmax(cosim_w * self.cfg.embed_temp, 2)
            return self.packer_w.new(probs=prob_w, embeds=embed_out_w)
        else:
            prob_w = F.log_softmax(self.linear_w(output_w), 2)
            return self.packer_w.new(probs=prob_w)
        #_, id_w = torch.max(cosim_w, 2)



    def _decode_free_run(self, code_w, max_len):
        code_w = code_w.unsqueeze(1)
        batch_size = code_w.size(0)

        # <sos>
        sos_w = self._get_sos_batch(batch_size, self.vocab_w)
        embed_in_w = self.embed_w(sos_w)
        # sos_embedding : [batch_size, 1, embedding_size]
        state_w = self._init_hidden(batch_size, self.cfg.hidden_size_w)

        # unroll
        if self.cfg.dec_embed:
            all_embed_w = []  # for differentiable input of discriminator
        all_prob_w = []  # for grad norm scaling
        all_id_w = []

        finished = torch.ByteTensor(batch_size, 1).zero_()
        finished = to_gpu(self.cfg.cuda,
                          Variable(finished, requires_grad=False))

        for i in range(max_len + 1):  # for each step
            # Decoder
            input_w = torch.cat([embed_in_w, code_w], 2)
            output_w, state_w = self.decoder(input_w, state_w)
            if self.cfg.dec_embed:
                embed_out_w = self.linear_w(output_w)
                cosim_w = self._compute_cosine_sim(embed_out_w, self.embed_w.embed)
                prob_w = F.log_softmax(cosim_w * self.cfg.embed_temp, 2)
                _, id_w = torch.max(cosim_w, 2)
                # if eos token has already appeared, fill zeros
                id_w, embed_out_w, finished = \
                    self._pads_after_eos(id_w, embed_out_w, finished)
            else:
                prob_w = F.log_softmax(self.linear_w(output_w), 2)
                _, id_w = torch.max(prob_w, 2)
                id_w, finished = self._pad_ids_after_eos(id_w, finished)
            # NOTE : words_prob is not considered here

            embed_in_w = self.embed_w(id_w)
            #embed_in_w = embed_out_w

            # append generated token ids & outs at each step
            if self.cfg.dec_embed:
                all_embed_w.append(embed_out_w)
            all_prob_w.append(prob_w)
            all_id_w.append(id_w)

        # concatenate all the results
        # words_id = torch.cat(all_words_id, 1)
        if self.cfg.dec_embed:
            embed_w = torch.cat(all_embed_w, 1)
        prob_w = torch.cat(all_prob_w, 1)
        id_w = torch.cat(all_id_w, 1)

        if self.cfg.dec_embed:
            return self.packer_w.new(probs=prob_w, ids=id_w, embeds=embed_w)
        else:
            return self.packer_w.new(probs=prob_w, ids=id_w)

    def _init_weights(self):
        # unifrom initialization in the range of [-0.1, 0.1]
        initrange = 0.1

        for p in self.decoder.parameters():
            p.data.uniform_(-initrange, initrange)
        self.linear_w.weight.data.uniform_(-initrange, initrange)
        self.linear_w.bias.data.fill_(0)

        if self.cfg.pos_tag:
            for p in self.tagger.parameters():
                p.data.uniform_(-initrange, initrange)
            self.linear_t.weight.data.uniform_(-initrange, initrange)
            self.linear_t.bias.data.fill_(0)

    def _init_hidden(self, bsz, nhidden):
        nlayers = self.cfg.nlayers
        zeros1 = Variable(torch.zeros(nlayers, bsz, nhidden))
        zeros2 = Variable(torch.zeros(nlayers, bsz, nhidden))
        return (to_gpu(self.cfg.cuda, zeros1), to_gpu(self.cfg.cuda, zeros2))

    def _init_state(self, bsz):
        zeros = Variable(torch.zeros(self.cfg.nlayers, bsz, self.cfg.nhidden))
        return to_gpu(self.cfg.cuda, zeros)

    def _get_sos_batch(self, bsz, vocab):
        sos_ids = to_gpu(self.cfg.cuda, Variable(torch.ones(bsz, 1).long()))
        sos_ids.fill_(vocab.SOS_ID)
        return sos_ids

    def _get_tag_batch(self, size, num):
        return to_gpu(self.cfg.cuda, Variable(torch.ones(*size, 1))) * num

    def _pads_after_eos(self, ids, out, finished):
        # zero ids after eos
        assert(self.vocab_w.PAD_ID == 0)
        finished = finished + ids.eq(self.vocab_w.EOS_ID).byte()
        ids_mask = finished.eq(0)
        ids = ids * ids_mask.long()
        out_mask = ids_mask.unsqueeze(2).expand_as(out)
        out = out * out_mask.float()
        return ids, out, finished

    def _pad_ids_after_eos(self, ids, finished):
        # zero ids after eos
        assert(self.vocab_w.PAD_ID == 0)
        finished = finished + ids.eq(self.vocab_w.EOS_ID).byte()
        ids_mask = finished.eq(0)
        ids = ids * ids_mask.long()
        return ids, finished

    def _compute_cosine_sim(self, out_embed, ref_embed):
        # compute cosine similarity
        ref_embed = F.normalize(ref_embed.weight, p=2, dim=1).detach()
        vocab_size, embed_size = ref_embed.size()
        ref_embed = ref_embed.permute(1, 0)  # [embed_size, vocab_size]
        out_embed_size = out_embed.size()
        # [bsz(*max_len), embed_size]
        out_embed = out_embed.view(-1, embed_size)
        cos_sim = torch.mm(out_embed, ref_embed)  # [bsz(*max_len), vocab_size]
        cos_sim = cos_sim.view(*out_embed_size[:-1], vocab_size)
        return cos_sim  # [bsz, (max_len,) vocab_size]


class DecoderCNN(BaseDecoder):
    def __init__(self, cfg, embed):
        super(DecoderCNN, self).__init__()
        # expected input dim
        #   : [bsz, c(embed or hidden size), h(1), w(max_len)]
        # main convoutional layers
        arch = cfg.arch_cnn
        self.cfg = cfg
        self.vocab = embed.vocab
        self.result_packer = DecoderOutPackerCNN(self, embed)
        self.deconvs = []
        for i in reversed(range(arch.n_conv)):
            deconv = nn.ConvTranspose2d(arch.c[i + 1], arch.c[i],
                                        (1, arch.f[i]), arch.s[i])
            self.deconvs.append(deconv)
            self.add_module("Deconv(%d)" % (i + 1), deconv)

        self.criterion_ce = nn.CrossEntropyLoss()
        self.criterion_nll = nn.NLLLoss()

    def _decode(self, inpack):
        code = inpack.code
        # NOTE : lengths can be used for pad masking

        x = code.view(*code.size(), 1, 1)
        for i, deconv in enumerate(self.deconvs):
            x = deconv(x)

        embed = x.squeeze().permute(0, 2, 1)
        embed = F.normalize(embed, p=2, dim=2)

        assert len(embed.size()) == 3
        assert embed.size(1) == self.cfg.max_len
        assert embed.size(2) == self.cfg.embed_size_w

        return self.result_packer.new(embed)  # [bsz, hidden_size]


class DecoderOutPackerRNN(object):
    def __init__(self, cfg, vocab):
        self.cfg = cfg
        self.vocab = vocab

    def new(self, probs, ids=None, embeds=None):
        return DecoderOutPack(self, probs, ids, embeds)


class DecoderOutPackerCNN(object):
    def __init__(self, decoder, embed_ref):
        self.cfg = decoder.cfg
        self.vocab = decoder.vocab
        self.embed_ref = embed_ref

    def new(self, embeds):
        cosim = self._compute_cosine_sim(embeds)
        _, ids = torch.max(cosim, dim=2)
        probs = F.log_softmax(cosim * self.cfg.embed_temp, 2)
        #probs = probs.view(-1, len(self.vocab))
        return DecoderOutPack(self, embeds, probs, ids)

    def _compute_cosine_sim(self, out_embed):
        # compute cosine similarity
        embed = F.normalize(self.embed_ref.embed.weight,
                            p=2, dim=1)  # .detach()
        embed = Variable(embed.data, requires_grad=False)
        vocab_size, embed_size = embed.size()
        embed = embed.permute(1, 0)  # [embed_size, vocab_size]
        out_size = out_embed.size()
        # [bsz(*max_len), embed_size]
        out_embed = out_embed.view(-1, embed_size)
        cos_sim = torch.mm(out_embed, embed)  # [bsz(*max_len), vocab_size]
        cos_sim = cos_sim.view(*out_size[:-1], vocab_size)
        return cos_sim  # [bsz, (max_len,) vocab_size]


class DecoderOutPack(object):
    def __init__(self, packer, probs, ids=None, embeds=None):
        self.cfg = packer.cfg
        self.vocab = packer.vocab
        self.embed = embeds
        if ids is None:
            _, ids = torch.max(probs, dim=2)
        self.id = WordIdTranscriber(ids, packer.vocab)
        self.prob = probs

    def get_text(self, num_sample=None):
        if num_sample is None:
            num_sample = self.cfg.log_nsample
        return self.id.to_text(num_sample)

    def get_text_with_pair(self, id_target, num_sample=None):
        if num_sample is None:
            num_sample = self.cfg.log_nsample
        return self.id.to_text_with_pair(id_target, num_sample)

    def get_text_batch(self):
        return self.id.to_text_batch()


class WordIdTranscriber(object):
    def __init__(self, ids, vocab):
        self.vocab = vocab
        self.ids = ids.data.cpu().numpy()

    def to_text_batch(self):
        return self.vocab.ids2text_batch(self.ids)

    def to_text(self, num_sample):
        ids_batch = self.vocab.ids2words_batch(self.ids)
        # np.random.shuffle(ids_batch)
        out_str = ''
        for i, ids in enumerate(ids_batch):
            out_str += self._get_line()
            if i >= num_sample:
                break
            out_str += ' '.join(ids) + '  \n'
            # we need double space here for markdown linebreak
        return out_str

    def to_text_with_pair(self, ids_pair, num_sample):
        ids_pair = ids_pair.data.cpu().numpy()
        ids_batch_x = self.vocab.ids2words_batch(ids_pair)
        ids_batch_y = self.vocab.ids2words_batch(self.ids)
        coupled = list(zip(ids_batch_x, ids_batch_y))
        np.random.shuffle(coupled)
        out_str = ''
        for i, (ids_x, ids_y) in enumerate(coupled):
            out_str += self._get_line()
            if i >= num_sample:
                break
            out_str += '[X]' + ' '.join(ids_x) + '  \n'
            out_str += '[Y]' + ' '.join(ids_y) + '  \n'
            # we need double space here for markdown linebreak
        return out_str

    def _get_line(self, char='-', row=1, length=130):
        out_str = ''
        for i in range(row):
            out_str += char * length
        return out_str + '\n'
