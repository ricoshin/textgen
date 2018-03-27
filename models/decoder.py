import logging

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.base_module import BaseDecoder
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from utils.utils import to_gpu
from utils.writer import ResultWriter

log = logging.getLogger('main')


class DecoderRNN(BaseDecoder):
    def __init__(self, cfg, embed_w, embed_t):
        super(DecoderRNN, self).__init__()
        self.cfg = cfg
        self.embed_t = embed_t
        self.embed_w = embed_w
        self.vocab_t = embed_t.vocab
        self.vocab_w = embed_w.vocab
        self.packer_t = ResultPackerRNN(cfg, embed_t.vocab)
        self.packer_w = ResultPackerRNN(cfg, embed_w.vocab)

        if cfg.pos_tag:
            input_size = cfg.embed_size_t + cfg.hidden_size_t
            self.tagger = nn.LSTM(input_size=input_size,
                                  hidden_size=cfg.hidden_size_t,
                                  dropout=cfg.dropout,
                                  batch_first=True)
            self.linear_t = nn.Linear(cfg.hidden_size_t, cfg.embed_size_t)

        input_size = cfg.embed_size_t + cfg.embed_size_w + cfg.hidden_size_w
        self.decoder = nn.LSTM(input_size=input_size,
                               hidden_size=cfg.hidden_size_w,
                               num_layers=1,
                               dropout=cfg.dropout,
                               batch_first=True)
        self.linear_w = nn.Linear(cfg.hidden_size_w, cfg.embed_size_w)

        self.criterion_ce = nn.CrossEntropyLoss()
        self.criterion_nll = nn.NLLLoss()
        self._init_weights()

    def teacher_forcing(self, code_tag, code_dec, batch):
        return self.__call__(
            code_tag, code_dec, batch.src_tag, batch.src, batch.len, mode='tf')

    def free_running(self, code_t, code_w, max_len):
        return self.__call__(code_t, code_w, max_len, mode='fr')

    def forward(self, *inputs, mode):
        if mode == 'tf':  # teacher forcing
            # [code, batch.src, batch.len]
            decoded = self._decode_tf(*inputs)
        elif mode == 'fr':  # free running
            # [code, max_len]
            decoded = self._decode_fr(*inputs)
        else:
            raise Exception("Unknown decoding mode!")

        return decoded

    def _decode_tf(self, code_t, code_w, tags, words, lengths):
        batch_size = code_t.size(0)

        # insert sos in the first column
        sos_t = self._get_sos_batch(batch_size, self.vocab_w)
        sos_w = self._get_sos_batch(batch_size, self.vocab_t)
        tags = torch.cat([sos_t, tags], 1)
        words = torch.cat([sos_w, words], 1)

        # length should be increased as well
        lengths = [length + 1 for length in lengths]

        init_state_t = self._init_hidden(batch_size, self.cfg.hidden_size_t)
        init_state_w = self._init_hidden(batch_size, self.cfg.hidden_size_w)

        embed_in_t = self.embed_t(tags)  # for teacher forcing
        embed_in_w = self.embed_w(words)

        all_code_t = code_t.unsqueeze(1).repeat(1, max(lengths), 1)
        all_code_w = code_w.unsqueeze(1).repeat(1, max(lengths), 1)

        # POS tagger
        input_t = torch.cat([embed_in_t, all_code_t], 2)
        packed_input_t = pack_padded_sequence(input=input_t,
                                              lengths=lengths,
                                              batch_first=True)
        packed_output_t, _ = self.tagger(packed_input_t, init_state_t)
        output_t, _ = pad_packed_sequence(packed_output_t, batch_first=True)

        embed_out_t = self.linear_t(output_t)
        cosim_t = self._compute_cosine_sim(embed_out_t, self.embed_t.embed)
        prob_t = F.log_softmax(cosim_t * self.cfg.embed_temp, 2)
        _, id_t = torch.max(cosim_t, 2)
        embed_out_t = self.embed_t(id_t)

        # Decoder
        embed_out_t_d = Variable(embed_out_t.data, requires_grad=False)
        input_w = torch.cat([embed_out_t_d, embed_in_w, all_code_w], 2)
        packed_input_w = pack_padded_sequence(input=input_w,
                                              lengths=lengths,
                                              batch_first=True)
        packed_output_w, _ = self.decoder(packed_input_w, init_state_w)
        output_w, length_w = pad_packed_sequence(
            packed_output_w, batch_first=True)

        # output layer for word prediction
        embed_out_w = self.linear_w(output_w)
        cosim_w = self._compute_cosine_sim(embed_out_w, self.embed_w.embed)
        prob_w = F.log_softmax(cosim_w * self.cfg.embed_temp, 2)
        #_, id_w = torch.max(cosim_w, 2)

        result_t = self.packer_t.new(embed_out_t, prob_t)
        result_w = self.packer_w.new(embed_out_w, prob_w)

        return result_t, result_w

    def _decode_fr(self, code_t, code_w, max_len):
        code_t = code_t.unsqueeze(1)
        code_w = code_w.unsqueeze(1)
        batch_size = code_t.size(0)

        # <sos>
        sos_t = self._get_sos_batch(batch_size, self.vocab_w)
        sos_w = self._get_sos_batch(batch_size, self.vocab_t)
        embed_in_t = self.embed_t(sos_t)
        embed_in_w = self.embed_w(sos_w)
        # sos_embedding : [batch_size, 1, embedding_size]

        state_t = self._init_hidden(batch_size, self.cfg.hidden_size_t)
        state_w = self._init_hidden(batch_size, self.cfg.hidden_size_w)

        # unroll
        all_embed_t = []  # for differentiable input of discriminator
        all_prob_t = []
        all_embed_w = []
        all_prob_w = []  # for grad norm scaling

        finished = torch.ByteTensor(batch_size, 1).zero_()
        finished = to_gpu(self.cfg.cuda,
                          Variable(finished, requires_grad=False))

        for i in range(max_len + 1):  # for each step
            # POS tagger
            input_t = torch.cat([embed_in_t, code_t], 2)
            output_t, state_t = self.tagger(input_t, state_t)
            embed_out_t = self.linear_t(output_t)
            cosim_t = self._compute_cosine_sim(embed_out_t, self.embed_t.embed)
            prob_t = F.log_softmax(cosim_t * self.cfg.embed_temp, 2)
            _, id_t = torch.max(cosim_t, 2)
            embed_in_t = self.embed_t(id_t)
            #embed_in_t = embed_out_t

            # Decoder
            embed_out_t_d = Variable(embed_out_t.data, requires_grad=False)
            input_w = torch.cat([embed_out_t_d, embed_in_w, code_w], 2)
            output_w, state_w = self.decoder(input_w, state_w)
            embed_out_w = self.linear_w(output_w)
            cosim_w = self._compute_cosine_sim(embed_out_w, self.embed_w.embed)
            prob_w = F.log_softmax(cosim_w * self.cfg.embed_temp, 2)
            _, id_w = torch.max(cosim_w, 2)
            embed_in_w = self.embed_w(id_w)
            #embed_in_w = embed_out_w

            # if eos token has already appeared, fill zeros
            # words_id, words_embed, finished = \
            #     self._pads_after_eos(words_id, words_embed, finished)
            # NOTE : words_prob is not considered here

            # append generated token ids & outs at each step
            all_embed_t.append(embed_out_t)
            all_prob_t.append(prob_t)
            all_embed_w.append(embed_out_w)
            all_prob_w.append(prob_w)
            #all_words_id.append(words_id)
            #all_tags.append(tags)

        # concatenate all the results
        embed_t = torch.cat(all_embed_t, 1)
        prob_t = torch.cat(all_prob_t, 1)
        # words_id = torch.cat(all_words_id, 1)
        embed_w = torch.cat(all_embed_w, 1)
        prob_w = torch.cat(all_prob_w, 1)

        result_t = self.packer_t.new(embed_out_t, prob_t)
        result_w = self.packer_w.new(embed_out_w, prob_w)

        return result_t, result_w

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
        assert(self.vocab.PAD_ID == 0)
        finished = finished + ids.eq(self.vocab.EOS_ID).byte()
        ids_mask = finished.eq(0)
        ids = ids * ids_mask.long()
        out_mask = ids_mask.unsqueeze(2).expand_as(out)
        out = out * out_mask.float()
        return ids, out, finished

    def _pad_ids_after_eos(self, ids, finished):
        # zero ids after eos
        assert(self.vocab.PAD_ID == 0)
        finished = finished + ids.eq(self.vocab.EOS_ID).byte()
        ids_mask = finished.eq(0)
        ids = ids * ids_mask.long()
        return ids, finished

    def _compute_cosine_sim(self, out_embed, ref_embed):
        # compute cosine similarity
        ref_embed = F.normalize(ref_embed.weight, p=2, dim=1).detach()
        vocab_size, embed_size = ref_embed.size()
        ref_embed = ref_embed.permute(1, 0)  # [embed_size, vocab_size]
        out_embed_size = out_embed.size()
        out_embed = out_embed.view(-1, embed_size)  # [bsz(*maxlen), embed_size]
        cos_sim = torch.mm(out_embed, ref_embed)  # [bsz(*maxlen), vocab_size]
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
        self.result_packer = ResultPackerCNN(self, embed)
        self.deconvs = []
        for i in reversed(range(arch.n_conv)):
            deconv = nn.ConvTranspose2d(arch.c[i + 1], arch.c[i],
                                        (1, arch.f[i]), arch.s[i])
            self.deconvs.append(deconv)
            self.add_module("Deconv(%d)" % (i + 1), deconv)

        self.criterion_ce = nn.CrossEntropyLoss()
        self.criterion_nll = nn.NLLLoss()

    def forward(self, code):
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


class ResultPackerRNN(object):
    def __init__(self, cfg, vocab):
        self.cfg = cfg
        self.vocab = vocab

    def new(self, embeds, probs, ids=None):
        return ResultPackage(self, embeds, probs, ids)


class ResultPackerCNN(object):
    def __init__(self, decoder, embed_ref):
        self.cfg = decoder.cfg
        self.vocab = decoder.vocab
        self.embed_ref = embed_ref

    def new(self, embeds):
        cosim = self._compute_cosine_sim(embeds)
        _, ids = torch.max(cosim, dim=2)
        probs = F.log_softmax(cosim * self.cfg.embed_temp, 2)
        return ResultPackage(self, embeds, ids, probs)

    def _compute_cosine_sim(self, out_embed):
        # compute cosine similarity
        embed = F.normalize(self.embed_ref.embed.weight,
                            p=2, dim=1)  # .detach()
        embed = Variable(embed.data, requires_grad=False)
        vocab_size, embed_size = embed.size()
        embed = embed.permute(1, 0)  # [embed_size, vocab_size]
        out_size = out_embed.size()
        # [bsz(*maxlen), embed_size]
        out_embed = out_embed.view(-1, embed_size)
        cos_sim = torch.mm(out_embed, embed)  # [bsz(*maxlen), vocab_size]
        cos_sim = cos_sim.view(*out_size[:-1], vocab_size)
        return cos_sim  # [bsz, (max_len,) vocab_size]


class ResultPackage(object):
    def __init__(self, packer, embeds, probs, ids=None):
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
        #np.random.shuffle(ids_batch)
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
