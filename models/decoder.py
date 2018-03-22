import logging
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from models.base_module import BaseDecoder
from utils.writer import ResultWriter
from utils.utils import to_gpu

log = logging.getLogger('main')


class DecoderRNN(BaseDecoder):
    def __init__(self, cfg, embed):
        super(DecoderRNN, self).__init__()
        self.cfg = cfg
        self.embed = embed
        self.vocab = embed.vocab
        self.result_packer = ResultPackerRNN(self)

        input_size_dec = cfg.word_embed_size + cfg.hidden_size
        self.decoder = nn.LSTM(input_size=input_size_dec,
                               hidden_size=cfg.hidden_size,
                               num_layers=1,
                               dropout=cfg.dropout,
                               batch_first=True)

        self.linear_word = nn.Linear(cfg.hidden_size, cfg.word_embed_size)

        if cfg.pos_tag:
            input_size_tag = cfg.word_embed_size + cfg.hidden_size
            self.tagger = nn.LSTM(input_size=input_size_tag,
                                  hidden_size=cfg.hidden_size,
                                  dropout=cfg.dropout,
                                  batch_first=True)
            self.linear_tag = nn.Linear(cfg.hidden_size, cfg.tag_size)

        self.criterion_ce = nn.CrossEntropyLoss()
        self.criterion_nll = nn.NLLLoss()
        self._init_weights()

    def teacher_forcing(self, code, batch):
        return self.__call__(code, batch.src, batch.len, mode='tf')

    def free_running(self, code, max_len):
        return self.__call__(code, max_len, mode='fr')

    def forward(self, *inputs, mode):
        if mode == 'tf': # teacher forcing
            # [code, batch.src, batch.len]
            decoded = self._decode_tf(*inputs)
        elif mode == 'fr': # free running
            # [code, max_len]
            decoded = self._decode_fr(*inputs)
        else:
            raise Exception("Unknown decoding mode!")

        return decoded

    def _decode_tf(self, code, inputs, lengths):
        batch_size = inputs.size(0)

        # insert sos in the first column
        sos_ids = self._get_sos_batch(batch_size, self.vocab)
        inputs = torch.cat([sos_ids, inputs], 1)

        # length should be increased as well
        lengths = [length + 1 for length in lengths]

        # Decoder
        init_state = self._init_hidden(batch_size)
        embed_dec = self.embed(inputs) # for teacher forcing

        all_hidden = code.unsqueeze(1).repeat(1, max(lengths), 1)
        augmented_input = torch.cat([embed_dec, all_hidden], 2)
        packed_in_dec = pack_padded_sequence(input=augmented_input,
                                             lengths=lengths,
                                             batch_first=True)
        packed_out_dec, _ = self.decoder(packed_in_dec, init_state)
        out_dec, len_dec = pad_packed_sequence(packed_out_dec, batch_first=True)

        out_dec = out_dec.contiguous().view(-1, self.cfg.hidden_size)
        words_embed = self.linear_word(out_dec) # output layer for word prediction
        words_embed = words_embed.view(batch_size, max(lengths),
                                       self.cfg.word_embed_size)
        words_embed = F.normalize(words_embed, p=2, dim=2)
        words_cosim = self._compute_cosine_sim(words_embed)
        _, words_id = torch.max(words_cosim, 2)
        words_prob = F.log_softmax(words_cosim * self.cfg.embed_temp, 2)

        # POS tagger
        #new_out_dec = Variable(words.data, requires_grad=False)
        augmented_input = torch.cat([words_embed, all_hidden], 2)
        packed_in_tag = pack_padded_sequence(input=augmented_input,
                                             lengths=lengths,
                                             batch_first=True)

        packed_out_tag, _ = self.tagger(packed_in_tag, init_state)
        out_tag, _ = pad_packed_sequence(packed_out_tag, batch_first=True)

        # reshape to batch_size*maxlen x nhidden before linear over vocab


        out_tag = out_tag.contiguous().view(-1, self.cfg.hidden_size)
        tags = self.linear_tag(out_tag)
        tags = tags.view(batch_size, max(lengths), self.cfg.tag_size)

        return self.result_packer.new(words_embed, words_id, words_prob)

    def _decode_fr(self, code, max_len):
        batch_size = code.size(0)
        code = code.unsqueeze(1)
        state_dec = state_tag = self._init_hidden(batch_size)

        # <sos>
        sos_ids = self._get_sos_batch(batch_size, self.vocab)
        embed_dec = self.embed(sos_ids)
        # sos_embedding : [batch_size, 1, embedding_size]

        # unroll
        all_words_embed = [] # for differentiable input of discriminator
        all_words_prob = []
        all_words_id = []
        all_tags = [] # for grad norm scaling

        finished = torch.ByteTensor(batch_size, 1).zero_()
        finished = to_gpu(self.cfg.cuda,
                          Variable(finished, requires_grad=False))

        for i in range(max_len + 1): # for each step
            # decoder
            input_dec = torch.cat([embed_dec, code], 2)
            outs_dec, state_dec = self.decoder(input_dec, state_dec)
            words_embed = self.linear_word(outs_dec)
            words_embed = F.normalize(words_embed, p=2, dim=2)
            words_cosim = self._compute_cosine_sim(words_embed)
            words_prob = F.log_softmax(words_cosim * self.cfg.embed_temp, 2)
            _, words_id = torch.max(words_cosim, 2)

            # if eos token has already appeared, fill zeros
            words_id, words_embed, finished = \
                self._pads_after_eos(words_id, words_embed, finished)
            # NOTE : words_prob is not considered here

            embed_dec = self.embed(words_id)
            #embed_dec = words

            # tagger
            #words_detached = Variable(outs_dec.data, requires_grad=False)
            input_tag = torch.cat([words_embed, code], 2)
            outs_tag, state_tag = self.tagger(input_tag, state_tag)
            tags = self.linear_tag(outs_tag)

            # append generated token ids & outs at each step
            all_words_embed.append(words_embed)
            all_words_prob.append(words_prob)
            all_words_id.append(words_id)
            all_tags.append(tags)

        # concatenate all the results
        words_embed = torch.cat(all_words_embed, 1)
        words_prob = torch.cat(all_words_prob, 1)
        words_id = torch.cat(all_words_id, 1)
        tags = torch.cat(all_tags, 1)

        return self.result_packer.new(words_embed, words_id, words_prob)

    def _init_weights(self):
        # unifrom initialization in the range of [-0.1, 0.1]
        initrange = 0.1

        for p in self.decoder.parameters():
            p.data.uniform_(-initrange, initrange)
        self.linear_word.weight.data.uniform_(-initrange, initrange)
        self.linear_word.bias.data.fill_(0)

        if self.cfg.pos_tag:
            for p in self.tagger.parameters():
                p.data.uniform_(-initrange, initrange)
            self.linear_tag.weight.data.uniform_(-initrange, initrange)
            self.linear_tag.bias.data.fill_(0)

    def _init_hidden(self, bsz):
        nlayers = self.cfg.nlayers
        nhidden = self.cfg.hidden_size
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

    def _compute_cosine_sim(self, out_word):
        # compute cosine similarity
        embed = F.normalize(self.embed.embed.weight, p=2, dim=1).detach()
        vocab_size, embed_size = embed.size()
        embed = embed.permute(1, 0) # [embed_size, vocab_size]
        out_embed = out_word.view(-1, embed_size) # [bsz(*maxlen), embed_size]
        cos_sim = torch.mm(out_embed, embed) # [bsz(*maxlen), vocab_size]
        cos_sim = cos_sim.view(*out_word.size()[:-1], vocab_size)
        return cos_sim # [bsz, (max_len,) vocab_size]


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
            deconv = nn.ConvTranspose2d(arch.c[i+1], arch.c[i],
                                        (1, arch.f[i]), arch.s[i])
            self.deconvs.append(deconv)
            self.add_module("Deconv(%d)" % (i+1), deconv)

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
        assert embed.size(2) == self.cfg.word_embed_size

        return self.result_packer.new(embed) # [bsz, hidden_size]


class ResultPackerRNN(object):
    def __init__(self, decoder):
        self.cfg = decoder.cfg
        self.vocab = decoder.vocab

    def new(self, embeds, ids, probs):
        return ResultPackage(self, embeds, ids, probs)


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
        embed = F.normalize(self.embed_ref.embed.weight, p=2, dim=1)#.detach()
        embed = Variable(embed.data, requires_grad=False)
        vocab_size, embed_size = embed.size()
        embed = embed.permute(1, 0) # [embed_size, vocab_size]
        out_size = out_embed.size()
        out_embed = out_embed.view(-1, embed_size) # [bsz(*maxlen), embed_size]
        cos_sim = torch.mm(out_embed, embed) # [bsz(*maxlen), vocab_size]
        cos_sim = cos_sim.view(*out_size[:-1], vocab_size)
        return cos_sim # [bsz, (max_len,) vocab_size]


class ResultPackage(object):
    def __init__(self, packer, embeds, ids, probs):
        self.cfg = packer.cfg
        self.vocab = packer.vocab
        self.embed = embeds
        self.id = WordIdTranscriber(ids, packer.vocab)
        self.prob = probs

    def get_text(self, num_sample=None):
        if num_sample is None:
            num_sample = self.cfg.log_nsample
        return self.id.to_text(num_sample)

    def get_text_with_target(self, id_target, num_sample=None):
        if num_sample is None:
            num_sample = self.cfg.log_nsample
        return self.id.to_text_with_target(id_target, num_sample)

    def get_text_batch(self):
        return self.id.to_text_batch()


class WordIdTranscriber(object):
    def __init__(self, ids, vocab):
        self.vocab = vocab
        self.ids = ids.data.cpu().numpy()

    def to_text(self, num_sample=None):
        if num_sample is None:
            num_sample = self.cfg.log_nsample
        if self._is_autoencoded:
            return self._get_autoencoded_n_text_samples(num_sample)
        else:
            return self._get_decoded_n_text_samples(num_sample)

    def to_text_batch(self):
        return self.vocab.ids2text_batch(self.ids)

    def to_text(self, num_sample):
        ids_batch = self.vocab.ids2words_batch(self.ids)
        np.random.shuffle(ids_batch)
        out_str = ''
        for i, ids in enumerate(ids_batch):
            out_str += self._get_line()
            if i >= num_sample: break
            out_str += ' '.join(ids) + '  \n'
            # we need double space here for markdown linebreak
        return out_str

    def to_text_with_target(self, ids_target, num_sample):
        ids_target = ids_target.data.cpu().numpy()
        ids_batch_x = self.vocab.ids2words_batch(ids_target)
        ids_batch_y = self.vocab.ids2words_batch(self.ids)
        coupled = list(zip(ids_batch_x, ids_batch_y))
        np.random.shuffle(coupled)
        out_str = ''
        for i, (ids_x, ids_y) in enumerate(coupled):
            out_str += self._get_line()
            if i >= num_sample: break
            out_str += '[X]' + ' '.join(ids_x) + '  \n'
            out_str += '[Y]' + ' '.join(ids_y) + '  \n'
            # we need double space here for markdown linebreak
        return out_str

    def _get_line(self, char='-', row=1, length=130):
        out_str = ''
        for i in range(row):
            out_str += char * length
        return out_str + '\n'
