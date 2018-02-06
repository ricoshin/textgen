import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from nn.embedding import WordEmbedding
from train.train_helper import ResultPackage
from utils.utils import to_gpu

log = logging.getLogger('main')


class Decoder(nn.Module):
    def __init__(self, cfg, vocab):
        super(Decoder, self).__init__()
        self.cfg = cfg
        self.grad_norm = None
        self.vocab = vocab
        self.embed_dec = WordEmbedding(vocab_size=cfg.vocab_size,
                                       embed_size=cfg.word_embed_size,
                                       fix_embed=cfg.fix_embed,
                                       init_embed=vocab.embed)

    def forward(self, *input):
        raise NotImplementedError

    def _store_grad_norm(self, grad):
        norm = torch.norm(grad, 2, 1)
        self.grad_norm = norm.detach().data.mean()
        return grad


class DecoderRNN(Decoder):
    def __init__(self, cfg, vocab):
        super(DecoderRNN, self).__init__(cfg, vocab)
        input_size_dec = cfg.word_embed_size + cfg.hidden_size
        self.decoder = nn.LSTM(input_size=input_size_dec,
                               hidden_size=cfg.hidden_size,
                               num_layers=1,
                               dropout=cfg.dropout,
                               batch_first=True)

        self.linear_word = nn.Linear(cfg.hidden_size, cfg.vocab_size)

        if cfg.pos_tag:
            input_size_tag = cfg.hidden_size * 2
            self.tagger = nn.LSTM(input_size=input_size_tag,
                                  hidden_size=cfg.hidden_size,
                                  dropout=cfg.dropout,
                                  batch_first=True)
            self.linear_tag = nn.Linear(cfg.hidden_size, cfg.tag_size)

        self.criterion_ce = nn.CrossEntropyLoss()
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

    def _pads_after_eos(self, ids, out, finished):
        # zero ids after eos
        assert(self.vocab.PAD_ID == 0)
        finished = finished + ids.eq(self.vocab.EOS_ID).byte()
        ids_mask = finished.eq(0)
        ids = ids * ids_mask.long()
        out_mask = ids_mask.unsqueeze(2).expand_as(out)
        out = out * out_mask.float()
        return ids, out, finished

    def forward(self, hidden, ids_dec, ids_tag, lengths):
        decoded = self._decode(hidden, ids_dec, ids_tag, lengths)

        return decoded

    def _decode(self, hidden, ids_dec, ids_tag, lengths):
        batch_size = ids_dec.size(0)

        # insert sos in the first column
        sos_ids_dec = self._get_sos_batch(batch_size, self.vocab)
        ids_dec = torch.cat([sos_ids_dec, ids_dec], 1)

        # length should be increased as well
        lengths = [length + 1 for length in lengths]

        # Decoder
        init_state = self._init_hidden(batch_size)
        embed_dec = self.embed_dec(ids_dec) # for teacher forcing

        all_hidden = hidden.unsqueeze(1).repeat(1, max(lengths), 1)
        augmented_input = torch.cat([embed_dec, all_hidden], 2)
        packed_in_dec = pack_padded_sequence(input=augmented_input,
                                             lengths=lengths,
                                             batch_first=True)
        packed_out_dec, _ = self.decoder(packed_in_dec, init_state)
        out_dec, len_dec = pad_packed_sequence(packed_out_dec, batch_first=True)

        if out_dec.requires_grad:
            out_dec.register_hook(self._store_grad_norm)
            #log.debug("Decoder gradient norm has been saved.")

        # POS tagger
        new_out_dec = Variable(out_dec.data, requires_grad=False)
        augmented_input = torch.cat([new_out_dec, all_hidden], 2)
        packed_in_tag = pack_padded_sequence(input=augmented_input,
                                             lengths=lengths,
                                             batch_first=True)

        packed_out_tag, _ = self.tagger(packed_in_tag, init_state)
        out_tag, _ = pad_packed_sequence(packed_out_tag, batch_first=True)

        # reshape to batch_size*maxlen x nhidden before linear over vocab
        out_dec = out_dec.contiguous().view(-1, self.cfg.hidden_size)
        words = self.linear_word(out_dec) # output layer for word prediction
        words = words.view(batch_size, max(lengths), self.cfg.vocab_size)

        out_tag = out_tag.contiguous().view(-1, self.cfg.hidden_size)
        tags = self.linear_tag(out_tag)
        tags = tags.view(batch_size, max(lengths), self.cfg.tag_size)

        return words, tags # decoded.size() : batch x max_len x ntokens [logits]

    def generate(self, hidden):
        if not self.cfg.backprop_gen:
            # should not backprop gradient to enc/gen
            hidden = hidden.detach()

        batch_size = hidden.size(0)
        hidden = hidden.unsqueeze(1)
        state_dec = state_tag = self._init_hidden(batch_size)

        # <sos>
        sos_ids_dec = self._get_sos_batch(batch_size, self.vocab)
        embed_dec = self.embed_dec(sos_ids_dec)
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
            input_dec = torch.cat([embed_dec, hidden], 2)
            outs_dec, state_dec = self.decoder(input_dec, state_dec)
            #outs_dec = Variable(outs_dec.data, requires_grad=False)
            input_tag = torch.cat([outs_dec, hidden], 2)
            outs_tag, state_tag = self.tagger(input_tag, state_tag)
            # output.size() : bath_size x 1(max_len) x nhidden
            # state.size() : 1(num_layer) x batch_size x nhidden
            logit_word = self.linear_word(outs_dec.squeeze(1))
            logit_tag = self.linear_tag(outs_tag.squeeze(1))
            # output.squeeze(1) : batch_size x nhidden (output layer)
            # words : batch_size x ntokens
            _, ids_word = torch.max(logit_word, 1, keepdim=True)
            _, ids_tag = torch.max(logit_tag, 1, keepdim=True)

            outs = F.softmax(logit_tag/1e-6, 1).unsqueeze(1)
            # [batch_size, 1 ntoken]

            # if eos token has already appeared, fill zeros
            ids_word, outs, finished = self._pads_after_eos(ids_word, outs,
                                                            finished)

            # for the next step
            embed_dec = self.embed_dec(ids_word)

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
