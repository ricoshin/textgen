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
        self.vocab = vocab
        self.grad_norm = None
        # word embedding
        self.embed = WordEmbedding(cfg, vocab.embed_mat)

    def forward(self, *input):
        raise NotImplementedError

    def _store_grad_norm(self, grad):
        norm = torch.norm(grad, 2, 1)
        self.grad_norm = norm.detach().data.mean()
        return grad


class DecoderRNN(Decoder):
    def __init__(self, cfg, vocab):
        super(DecoderRNN, self).__init__(cfg, vocab)
        self.start_symbols = to_gpu(
            cfg.cuda, Variable(torch.ones(10, 1).long()))



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

    def _init_weights(self):
        # unifrom initialization in the range of [-0.1, 0.1]
        initrange = 0.1

        # Initialize Encoder and Decoder Weights
        for p in self.decoder.parameters():
            p.data.uniform_(-initrange, initrange)

        # Initialize Linear Weight
        self.linear.weight.data.uniform_(-initrange, initrange)
        self.linear.bias.data.fill_(0)

    def _init_hidden(self, bsz):
        nlayers = self.cfg.nlayers
        nhidden = self.cfg.hidden_size
        zeros1 = Variable(torch.zeros(nlayers, bsz, nhidden))
        zeros2 = Variable(torch.zeros(nlayers, bsz, nhidden))
        return (to_gpu(self.cfg.cuda, zeros1), to_gpu(self.cfg.cuda, zeros2))

    def _init_state(self, bsz):
        zeros = Variable(torch.zeros(self.cfg.nlayers, bsz, self.cfg.nhidden))
        return to_gpu(self.cfg.cuda, zeros)

    def _get_sos_batch(self, bsz):
        sos_ids = to_gpu(self.cfg.cuda, Variable(torch.ones(bsz, 1).long()))
        sos_ids.fill_(self.vocab.SOS_ID)
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

    def forward(self, hidden, indices, lengths):
        decoded = self._decode(hidden, indices, lengths)

        return decoded

    def _decode(self, hidden, indices, lengths):
        batch_size = indices.size(0)

        # insert sos in the first column
        sos_ids = self._get_sos_batch(batch_size)
        indices = torch.cat([sos_ids, indices], 1)
        # length should be increased as well
        lengths = [length + 1 for length in lengths]

        # hidden.size() : bsz x hidden_size
        # bsz x max_len x hidden_size
        all_hidden = hidden.unsqueeze(1).repeat(1, max(lengths), 1)

        if self.cfg.hidden_init:
            state = (hidden.unsqueeze(0), self._init_state(batch_size))
        else:
            state = self._init_hidden(batch_size)

        embeddings = self.embed(indices) # for teacher forcing
        try:
            augmented_embeddings = torch.cat([embeddings, all_hidden], 2)
        except:
            import ipdb; ipdb.set_trace()
        packed_embeddings = pack_padded_sequence(input=augmented_embeddings,
                                                 lengths=lengths,
                                                 batch_first=True)

        packed_output, state = self.decoder(packed_embeddings, state)
        output, lengths = pad_packed_sequence(packed_output, batch_first=True)

        if output.requires_grad:
            output.register_hook(self._store_grad_norm)
            #log.debug("Decoder gradient norm has been saved.")

        # reshape to batch_size*maxlen x nhidden before linear over vocab
        output = output.contiguous().view(-1, self.cfg.hidden_size)
        decoded = self.linear(output) # output layer
        decoded = decoded.view(batch_size, max(lengths), self.cfg.vocab_size)

        return decoded # decoded.size() : batch x max_len x ntokens [logits]

    def generate(self, hidden):
        if not self.cfg.backprop_gen:
            # should not backprop gradient to enc/gen
            hidden = hidden.detach()

        batch_size = hidden.size(0)

        if self.cfg.hidden_init:
            # initialize decoder hidden state to encoder output
            state = (hidden.unsqueeze(0), self._init_state(batch_size))
        else:
            state = self._init_hidden(batch_size)

        # <sos>
        sos_embedding = self.embed(self._get_sos_batch(batch_size))
        # sos_embedding : [batch_size, 1, embedding_size]
        inputs = torch.cat([sos_embedding, hidden.unsqueeze(1)], 2)

        # unroll
        all_ids = []
        all_outs = []
        all_logits = []

        finished = torch.ByteTensor(batch_size, 1).zero_()
        finished = to_gpu(self.cfg.cuda,
                          Variable(finished, requires_grad=False))
        max_len = self.cfg.max_len #+ 1 # including sos/eos
        for i in range(max_len): # for each step
            # input very first step (sos totken + fake_z_code)
            outs, state = self.decoder(inputs, state)
            # output.size() : bath_size x 1(max_len) x nhidden
            # state.size() : 1(num_layer) x batch_size x nhidden
            logits = self.linear(outs.squeeze(1))
            # output.squeeze(1) : batch_size x nhidden (output layer)
            # logits : batch_size x ntokens
            _, ids = torch.max(logits, 1, keepdim=True)

            if self.cfg.disc_s_in == 'embed':
                outs = F.softmax(logits/1e-6, 1).unsqueeze(1)
                # [batch_size, 1 ntoken]

            # if eos token has already appeared, fill zeros
            ids, outs, finished = self._pads_after_eos(ids, outs, finished)
            # append generated token ids & outs at each step

            # for the next step
            embed = self.embed(ids)
            inputs = torch.cat([embed, hidden.unsqueeze(1)], 2)

            all_ids.append(ids)
            all_outs.append(outs)
            all_logits.append(logits.unsqueeze(1))

        # concatenate all the results
        max_ids = torch.cat(all_ids, 1)
        max_ids = max_ids.data.cpu().numpy()
        outs = torch.cat(all_outs, 1)
        self.logits = torch.cat(all_logits, 1) # [bsz, max_len]

        # max_ids.size() : bsz x max_len
        # outs.size() : bsz x max_len X (vocab or hidden)_size
        return max_ids, outs
