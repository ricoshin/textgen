import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from train.train_helper import ResultPackage
from utils.utils import to_gpu

log = logging.getLogger('main')


class Decoder(nn.Module):
    def __init__(self, cfg, embed):
        super(Decoder, self).__init__()
        self.cfg = cfg
        self.grad_norm = None
        self.embed = embed
        self.vocab = embed.vocab

    def forward(self, *input):
        raise NotImplementedError

    def _store_grad_norm(self, grad):
        norm = torch.norm(grad, 2, 1)
        self.grad_norm = norm.detach().data.mean()
        return grad

TAG=True
TAG_DEC=0
TAG_GEN=0

class DecoderRNN(Decoder):
    def __init__(self, cfg, embed):
        super(DecoderRNN, self).__init__(cfg, embed)
        input_size_dec = cfg.word_embed_size + cfg.hidden_size + 1
        self.decoder = nn.LSTM(input_size=input_size_dec,
                               hidden_size=cfg.hidden_size,
                               num_layers=1,
                               dropout=cfg.dropout,
                               batch_first=True)

        self.linear_word = nn.Linear(cfg.hidden_size, cfg.word_embed_size)

        if cfg.pos_tag:
            input_size_tag = cfg.word_embed_size + cfg.hidden_size + 1
            self.tagger = nn.LSTM(input_size=input_size_tag,
                                  hidden_size=cfg.hidden_size,
                                  dropout=cfg.dropout,
                                  batch_first=True)
            self.linear_tag = nn.Linear(cfg.hidden_size, cfg.tag_size)

        self.criterion_ce = nn.CrossEntropyLoss()
        self.criterion_nll = nn.NLLLoss()
        self._init_weights()

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
        ids_mask = finished.eq(0)
        ids = ids * ids_mask.long()
        out_mask = ids_mask.unsqueeze(2).expand_as(out)
        out = out * out_mask.float()
        finished = finished + ids.eq(self.vocab.EOS_ID).byte()
        return ids, out, finished

    def _compute_cosine_sim(self, out_word):
        # compute cosine similarity
        embed = self.embed.embed.weight.detach()
        vocab_size, embed_size = embed.size()
        embed = embed.permute(1, 0) # [embed_size, vocab_size]
        out_embed = out_word.view(-1, embed_size) # [bsz(*maxlen), embed_size]
        cos_sim = torch.mm(out_embed, embed) # [bsz(*maxlen), vocab_size]
        cos_sim = cos_sim.view(*out_word.size()[:-1], vocab_size)
        return cos_sim # [bsz, (max_len,) vocab_size]

    def forward(self, *inputs, mode, save_grad_norm=False):
        if mode == 'tf': # teacher forcing
            decoded = self._decode_tf(*inputs)
        elif mode == 'fr': # free running
            decoded = self._decode_fr(*inputs, tag=TAG_DEC,
                                      save_grad_norm=save_grad_norm)
        elif mode == 'gen': # generator
            decoded = self._decode_fr(*inputs, tag=TAG_GEN,
                                      save_grad_norm=save_grad_norm)
        else:
            raise Exception("Unknown decoding mode!")
        return decoded

    def _decode_tf(self, hidden, ids_dec, lengths):
        batch_size = ids_dec.size(0)

        # insert sos in the first column
        sos_ids_dec = self._get_sos_batch(batch_size, self.vocab)
        ids_dec = torch.cat([sos_ids_dec, ids_dec], 1)

        # length should be increased as well
        lengths = [length + 1 for length in lengths]
        # generate tag
        mode_tag = self._get_tag_batch([batch_size, max(lengths)], TAG_DEC)

        # Decoder
        init_state = self._init_hidden(batch_size)
        embed_dec = self.embed(ids_dec) # for teacher forcing

        all_hidden = hidden.unsqueeze(1).repeat(1, max(lengths), 1)
        augmented_input = torch.cat([embed_dec, all_hidden, mode_tag], 2)
        packed_in_dec = pack_padded_sequence(input=augmented_input,
                                             lengths=lengths,
                                             batch_first=True)
        packed_out_dec, _ = self.decoder(packed_in_dec, init_state)
        out_dec, len_dec = pad_packed_sequence(packed_out_dec, batch_first=True)

        out_dec = out_dec.contiguous().view(-1, self.cfg.hidden_size)
        words = self.linear_word(out_dec) # output layer for word prediction
        words = words.view(batch_size, max(lengths), self.cfg.word_embed_size)
        words = F.normalize(words, p=2, dim=2)
        words_cos_sim = self._compute_cosine_sim(words)
        words_prob = F.log_softmax(words_cos_sim * self.cfg.embed_temp, 2)


        #if words_prob.requires_grad: # NOTE fix later!
        #    word_prob.register_hook(self._store_grad_norm)
            #log.debug("Decoder gradient norm has been saved.")

        # POS tagger
        #new_out_dec = Variable(words.data, requires_grad=False)
        augmented_input = torch.cat([words, all_hidden, mode_tag], 2)
        packed_in_tag = pack_padded_sequence(input=augmented_input,
                                             lengths=lengths,
                                             batch_first=True)

        packed_out_tag, _ = self.tagger(packed_in_tag, init_state)
        out_tag, _ = pad_packed_sequence(packed_out_tag, batch_first=True)

        # reshape to batch_size*maxlen x nhidden before linear over vocab


        out_tag = out_tag.contiguous().view(-1, self.cfg.hidden_size)
        tags = self.linear_tag(out_tag)
        tags = tags.view(batch_size, max(lengths), self.cfg.tag_size)

        return words, words_prob, tags

    def _decode_fr(self, hidden, lengths=None, tag=None, save_grad_norm=False):
        if lengths is None:
            max_len = self.cfg.max_len + 1
        else:
            max_len = max(lengths) + 1

        batch_size = hidden.size(0)
        hidden = hidden.unsqueeze(1)
        state_dec = state_tag = self._init_hidden(batch_size)

         # generate tag
        mode_tag = self._get_tag_batch([batch_size, 1], tag)

        # <sos>
        sos_ids_dec = self._get_sos_batch(batch_size, self.vocab)
        embed_dec = self.embed(sos_ids_dec)
        # sos_embedding : [batch_size, 1, embedding_size]

        # unroll
        all_words = [] # for differentiable input of discriminator
        all_words_prob = []
        all_tags = [] # for grad norm scaling

        finished = torch.ByteTensor(batch_size, 1).zero_()
        finished = to_gpu(self.cfg.cuda,
                          Variable(finished, requires_grad=False))

        for i in range(max_len): # for each step
            # decoder
            input_dec = torch.cat([embed_dec, hidden, mode_tag], 2)
            outs_dec, state_dec = self.decoder(input_dec, state_dec)
            words = self.linear_word(outs_dec)
            words = F.normalize(words, p=2, dim=2)
            words_cosim = self._compute_cosine_sim(words)
            words_prob = F.log_softmax(words_cosim * self.cfg.embed_temp, 2)
            _, words_id = torch.max(words_cosim, 2)
            embed_dec = self.embed(words_id)
            #embed_dec = words

            # tagger
            #words_detached = Variable(outs_dec.data, requires_grad=False)
            input_tag = torch.cat([words, hidden, mode_tag], 2)
            outs_tag, state_tag = self.tagger(input_tag, state_tag)
            tags = self.linear_tag(outs_tag)

            # append generated token ids & outs at each step
            all_words.append(words)
            all_words_prob.append(words_prob)
            all_tags.append(tags)

        # concatenate all the results
        words = torch.cat(all_words, 1)
        words_prob = torch.cat(all_words_prob, 1)
        tags = torch.cat(all_tags, 1)

        if words.requires_grad and save_grad_norm: # NOTE fix!!
            words.register_hook(self._store_grad_norm)

        return words, words_prob, tags

    def generate(self, hidden):
        if not self.cfg.backprop_gen:
            # should not backprop gradient to enc/gen
            hidden = hidden.detach()

        batch_size = hidden.size(0)
        hidden = hidden.unsqueeze(1)
        state_dec = state_tag = self._init_hidden(batch_size)

        # generate tag
        mode_tag = self._get_tag_batch([batch_size, 1], TAG_GEN)

        # <sos>
        sos_ids_dec = self._get_sos_batch(batch_size, self.vocab)
        embed_dec = self.embed(sos_ids_dec)
        # sos_embedding : [batch_size, 1, embedding_size]

        # unroll
        all_ids_word = []
        all_ids_tag = []
        all_outs = [] # for differentiable input of discriminator
        all_logits = [] # for grad norm scaling

        finished = torch.ByteTensor(batch_size, 1).zero_()
        finished = to_gpu(self.cfg.cuda,
                          Variable(finished, requires_grad=False))

        max_len = self.cfg.max_len #+ 1 # including sos/eos
        for i in range(max_len): # for each step

            # decoder
            input_dec = torch.cat([embed_dec, hidden, mode_tag], 2)
            outs_dec, state_dec = self.decoder(input_dec, state_dec)
            logit_word = self.linear_word(outs_dec)
            logit_word = F.normalize(logit_word, p=2, dim=2)
            cos_sim = self._compute_cosine_sim(logit_word)
            # for the next step
            _, ids_word = torch.max(cos_sim, 2)
            embed_dec = self.embed(ids_word)
            #embed_dec = logit_word

            # pos tagger
            #outs_dec = Variable(outs_dec.data, requires_grad=False)
            input_tag = torch.cat([logit_word, hidden, mode_tag], 2)
            outs_tag, state_tag = self.tagger(input_tag, state_tag)
            logit_tag = self.linear_tag(outs_tag)
            _, ids_tag = torch.max(logit_tag, 2)
            outs = F.softmax(logit_tag/1e-6, 1)
            # [batch_size, 1 ntoken]

            # if eos token has already appeared, fill zeros
            ids_word, outs, finished = self._pads_after_eos(ids_word, outs,
                                                            finished)

            # append generated token ids & outs at each step
            all_ids_word.append(ids_word)
            all_ids_tag.append(ids_tag)
            all_outs.append(outs)
            all_logits.append(logit_word.unsqueeze(1))

        # concatenate all the results
        max_ids_word = torch.cat(all_ids_word, 1)
        max_ids_word = max_ids_word.data.cpu().numpy()
        max_ids_tag = torch.cat(all_ids_tag, 1)
        max_ids_tag = max_ids_tag.data.cpu().numpy()

        outs = torch.cat(all_outs, 1)
        self.logits = torch.cat(all_logits, 1) # [bsz, max_len]

        # max_ids.size() : bsz x max_len
        # outs.size() : bsz x max_len X (vocab or hidden)_size
        return max_ids_word, max_ids_tag, outs
