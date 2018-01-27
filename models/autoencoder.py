import logging
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from train.train_helper import ResultPackage
from utils.utils import to_gpu

log = logging.getLogger('main')

class Encoder(nn.Module):
    def __init__(self, cfg, vocab):
        super(Encoder, self).__init__()
        self.cfg = cfg
        self.noise_radius = cfg.noise_radius
        # Vocabulary embedding
        self.embed = nn.Embedding(cfg.vocab_size, cfg.embed_size)
        self.embed.weight.data.copy_(torch.from_numpy(vocab.embed_mat))
        if cfg.load_glove and cfg.fix_embed:
            self.embed.weight.requires_grad = False
            
        # RNN Encoder and Decoder
        self.encoder = nn.LSTM(input_size=cfg.embed_size,
                               hidden_size=cfg.hidden_size,
                               num_layers=cfg.nlayers,
                               dropout=cfg.dropout,
                               batch_first=True)

        self._init_weights()

    def init_weights(self):
        # unifrom initialization in the range of [-0.1, 0.1]
        initrange = 0.1

        # Initialize Vocabulary Matrix Weight
        if not self.cfg.load_glove:
            self.embed.weight.data.uniform_(-initrange, initrange)
        # by default it's initialized with normal_(0,1)

        # Initialize Encoder and Decoder Weights
        for p in self.encoder.parameters():
            p.data.uniform_(-initrange, initrange)

    def store_grad_norm(self, grad):
        norm = torch.norm(grad, 2, 1)
        self.grad_norm = norm.detach().data.mean()
        # use this when compute disc_c's gradient (register_hook)
        return grad

    def forward(self, indices, lengths, noise, encode_only=False):
        batch_size, maxlen = indices.size()

        hidden = self._encode(indices, lengths, noise)

        if encode_only:
            return hidden # batch_size x hidden_size

        if hidden.requires_grad:
            hidden.register_hook(self.store_enc_grad_norm)

        return hidden

    def forward(self, indices, lengths, noise):
        # indices.size() : batch_size x max(lengths) [Variable]
        # len(lengths) : batch_size [List]
        embeddings = self.embed(indices)
        # embeddings.data.size() : batch_size x max(lenghts) x embed_dim [Variable]
        packed_embeddings = pack_padded_sequence(input=embeddings,
                                                 lengths=lengths,
                                                 batch_first=True)
        # Encode
        packed_output, state = self.encoder(packed_embeddings)
        hidden, cell = state # last states (tuple the length of 2)

        hidden = hidden[-1]  # get hidden state of last layer of encoder
        norms = torch.norm(hidden, 2, 1)

        hidden = torch.div(hidden, norms.unsqueeze(1).expand_as(hidden))

        if noise and self.cfg.noise_radius > 0:
            gauss_noise = torch.normal(means=torch.zeros(hidden.size()),
                                       std=self.noise_radius)
            hidden = hidden + to_gpu(self.cfg.cuda, Variable(gauss_noise))


            #log.debug("Encoder gradient norm has been saved.")

        return hidden # batch_size x hidden_size

    def encode_only(self, cfg, batch, train=True):
        if train:
            self.train()
            self.zero_grad()
        else:
            self.eval()
        # when encode_only is True:
        #   encoded hidden states (batch_size x hidden_size) are returned
        real_hidden = self(batch.src, batch.len, noise=False, encode_only=True)
        return real_hidden


class Decoder(nn.Module):
    def __init__(self, cfg, init_embed=None):
        super(Decoder, self).__init__()
        self.start_symbols = to_gpu(
            cfg.cuda, Variable(torch.ones(10, 1).long()))
        # Vocabulary embedding
        self.embed = nn.Embedding(cfg.vocab_size, cfg.embed_size)
        decoder_input_size = cfg.embed_size + cfg.hidden_size
        self.decoder = nn.LSTM(input_size=decoder_input_size,
                               hidden_size=cfg.hidden_size,
                               num_layers=1,
                               dropout=cfg.dropout,
                               batch_first=True)

        # Initialize Linear Transformation
        self.linear = nn.Linear(cfg.hidden_size, cfg.vocab_size)
        self.criterion_ce = nn.CrossEntropyLoss()
        self._init_weights()

    def init_weights(self):
        # unifrom initialization in the range of [-0.1, 0.1]
        initrange = 0.1

        # Initialize Vocabulary Matrix Weight
        if not self.cfg.load_glove:
            self.embed.weight.data.uniform_(-initrange, initrange)
        # by default it's initialized with normal_(0,1)

        # Initialize Encoder and Decoder Weights
        for p in self.decoder.parameters():
            p.data.uniform_(-initrange, initrange)

        # Initialize Linear Weight
        self.linear.weight.data.uniform_(-initrange, initrange)
        self.linear.bias.data.fill_(0)

    def init_hidden(self, bsz):
        nlayers = self.cfg.nlayers
        nhidden = self.cfg.hidden_size
        zeros1 = Variable(torch.zeros(nlayers, bsz, nhidden))
        zeros2 = Variable(torch.zeros(nlayers, bsz, nhidden))
        return (to_gpu(self.cfg.cuda, zeros1), to_gpu(self.cfg.cuda, zeros2))

    def init_state(self, bsz):
        zeros = Variable(torch.zeros(self.cfg.nlayers, bsz, self.cfg.nhidden))
        return to_gpu(self.cfg.cuda, zeros)


    def store_grad_norm(self, grad):
        norm = torch.norm(grad, 2, 1)
        self.grad_norm = norm.detach().data.mean()
        # use this when compute disc_s's gradient (register_hook)
        return grad


    def forward(self, indices, lengths, noise, encode_only=False):
        batch_size, maxlen = indices.size()

        decoded = self._decode(hidden, batch_size, maxlen,
                              indices=indices, lengths=lengths)

        return decoded

    def _decode(self, hidden, batch_size, maxlen, indices=None, lengths=None):
        # batch x hidden
        # hidden.size() : batch_size x hidden_size
        # hidden.unsqueeze(1) : batch_size x 1 x hidden_size
        # hidden.unsqueeze(1).repeat(1, maxlen, 1) : batch_size x max_len x hidden_size
        all_hidden = hidden.unsqueeze(1).repeat(1, maxlen, 1)

        if self.cfg.hidden_init:
            state = (hidden.unsqueeze(0), self.init_state(batch_size))
        else:
            state = self.init_hidden(batch_size)

        embeddings = self.embed_dec(indices) # for teacher forcing
        augmented_embeddings = torch.cat([embeddings, all_hidden], 2)
        packed_embeddings = pack_padded_sequence(input=augmented_embeddings,
                                                 lengths=lengths,
                                                 batch_first=True)

        packed_output, state = self.decoder(packed_embeddings, state)
        output, lengths = pad_packed_sequence(packed_output, batch_first=True)

        if output.requires_grad:
            output.register_hook(self.store_dec_grad_norm)
            #log.debug("Decoder gradient norm has been saved.")

        # reshape to batch_size*maxlen x nhidden before linear over vocab
        output = output.contiguous().view(-1, self.cfg.hidden_size)
        decoded = self.linear(output) # output layer
        decoded = decoded.view(batch_size, maxlen, self.cfg.vocab_size)

        return decoded # decoded.size() : batch x max_len x ntokens [logits]

    def decode_only(self, cfg, hidden, vocab, train=True):
        if train:
            self.train()
            self.zero_grad()
        else:
            self.eval()

        if not (train and cfg.backprop_gen):
            # should not backprop gradient to enc/gen
            hidden = hidden.detach()

        batch_size = hidden.size(0)

        if cfg.hidden_init:
            # initialize decoder hidden state to encoder output
            state = (hidden.unsqueeze(0), self.init_state(batch_size))
        else:
            state = self.init_hidden(batch_size)

        # <sos>
        self.start_symbols.data.resize_(batch_size, 1)
        self.start_symbols.data.fill_(1)
        # self.start_symbols.size() : [batch_size, 1]

        embedding = self.embed_dec(self.start_symbols)
        # embedding : [batch_size, 1, embedding_size]
        inputs = torch.cat([embedding, hidden.unsqueeze(1)], 2)

        # unroll
        all_ids = []
        all_outs = []

        eos_id = 2
        finished = torch.ByteTensor(batch_size, 1).zero_().cuda()
        finished = Variable(finished, requires_grad=False)
        max_len = cfg.max_len + 1 # including sos/eos
        for i in range(max_len): # for each step
            # input very first step (sos totken + fake_z_code)
            outs, state = self.decoder(inputs, state)
            # output.size() : bath_size x 1(max_len) x nhidden
            # state.size() : 1(num_layer) x batch_size x nhidden
            overvocab = self.linear(outs.squeeze(1))
            # output.squeeze(1) : batch_size x nhidden (output layer)
            # overvocab : batch_size x ntokens
            _, ids = torch.max(overvocab, 1, keepdim=True)

            if cfg.disc_s_in == 'embed':
                outs = F.softmax(overvocab/1e-6, 1).unsqueeze(1)
                # [batch_size, 1 ntoken]

            # if eos token has already appeared, fill zeros
            ids_mask = finished.eq(0).long()
            outs_mask = finished.eq(0).unsqueeze(2).expand_as(outs).float()
            ids = ids * ids_mask
            outs = outs * outs_mask
            finished += ids.eq(vocab.EOS_ID).byte()

            # append generated token ids & outs at each step
            all_ids.append(ids)
            all_outs.append(outs)
            # for the next step
            embed = self.embed_dec(ids)
            inputs = torch.cat([embed, hidden.unsqueeze(1)], 2)

        # concatenate all the results
        max_ids = torch.cat(all_ids, 1)
        max_ids = max_ids.data.cpu().numpy()
        outs = torch.cat(all_outs, 1)

        # max_ids.size() : bsz x max_len
        # outs.size() : bsz x max_len X (vocab or hidden)_size
        return max_ids, outs
