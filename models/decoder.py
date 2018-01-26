import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from nn.embedding import WordEmbedding
from utils.utils import to_gpu

log = logging.getLogger('main')


class Decoder(nn.Module):
    def __init__(self, cfg, vocab):
        super(Decoder, self).__init__()
        self.cfg = cfg
        self.vocab = vocab
        self.grad_norm = None
        self.embedding = WordEmbedding(cfg, vocab.embed_mat)

    def forward(self, *input):
        raise NotImplementedError

    def _check_train(self, train):
        if train:
            self.train()
            self.zero_grad()
        else:
            self.eval()

    def _store_grad_norm(self, grad):
        norm = torch.norm(grad, 2, 1)
        self.grad_norm = norm.detach().data.mean()
        return grad

    def _save_norm(self, logits):
        if logits.requires_grad:
            logits.register_hook(self._store_grad_norm)


class DecoderRNN(Decoder):
    def __init__(self, cfg, vocab):
        super(DecoderRNN, self).__init__(cfg, vocab)
        # RNN Decoder
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

    def _initial_state(self, code):
        batch_size, hidden_size = code.size()
        num_layers = self.cfg.nlayers

        def zero_state():
            zeros = torch.zeros(num_layers, batch_size, hidden_size)
            return to_gpu(self.cfg.cuda, Variable(zeros))

        if self.cfg.hidden_init:
            return (code.unsqueeze(0), zero_state())
        else:
            return (zero_state(), zero_state())

    def _get_sos_batch(self, bsz):
        sos_ids = to_gpu(self.cfg.cuda, Variable(torch.ones(bsz, 1).long()))
        sos_ids.fill_(self.vocab.SOS_ID)
        return self.embedding(sos_ids)

    def _pads_after_eos(self, ids, outs, finished):
        ids_mask = finished.eq(0)
        outs_mask = ids_mask.unsqueeze(2).expand_as(outs)
        ids = ids * ids_mask.long()
        outs = outs * outs_mask.float()
        finished += ids.eq(self.vocab.EOS_ID).byte()

    def forward(self, *args, ae_mode=False, train=False):
        self._check_train(train)

        if ae_mode: # autoencoder
            assert(len(args) == 3)
            outs = self._teacher_force(*args)
        else: # decode only
            assert(len(args) == 1)
            outs = self._free_run(*args)

        return outs

    def _teacher_force(self, code, indices, lengths):
        # code : [bsz, hidden_size]
        # indices : [bsz, max_len]
        assert(code.size(1) == self.cfg.hidden_size)
        batch_size, max_len = indices.size()
        init_state = self._initial_state(code)
        code_for_all_steps = code.unsqueeze(1).repeat(1, max_len, 1)
        # [batch_size x max_len x hidden_size]

        embeddings = self.embedding(indices) # for teacher forcing
        inputs = torch.cat([embeddings, code_for_all_steps], 2)
        packed_inputs = pack_padded_sequence(input=inputs,
                                             lengths=lengths,
                                             batch_first=True)

        packed_outputs, states = self.decoder(packed_inputs, init_state)
        outputs, lengths = pad_packed_sequence(packed_outputs, batch_first=True)


        # reshape to batch_size*maxlen x nhidden before linear over vocab
        outputs = outputs.contiguous().view(-1, self.cfg.hidden_size)
        logits = self.linear(outputs) # output layer
        logits = logits.view(batch_size, max_len, self.cfg.vocab_size)

        # save norm for later
        self._save_norm(logits)

        return logits # decoded.size() : batch x max_len x ntokens [logits]

    def _free_run(self, code):
        # cut the grpah NOTE : what about enc?
        #if not self.cfg.backprop_gen:
        #    code = code.detach()

        batch_size = code.size(0)
        init_state = self._initial_state(code)

        # sos_embedding : [batch_size, 1, embedding_size]
        sos_embedding = self._get_sos_batch(batch_size)
        input_ = torch.cat([sos_embedding, code.unsqueeze(1)], 2)

        # unroll
        all_ids = []
        all_outs = []
        all_logits = []

        finished = torch.ByteTensor(batch_size, 1).zero_()
        finished = to_gpu(self.cfg.cuda,
                          Variable(finished, requires_grad=False))

        max_len = self.cfg.max_len + 1 # including sos/eos
        for i in range(max_len): # for each step
            # input very first step (sos totken + fake_z_code)
            out, state = self.decoder(input_, init_state)
            # outs.size() : bath_size x 1(max_len) x nhidden
            # state.size() : 1(num_layer) x batch_size x nhidden
            logit = self.linear(out.squeeze(1)) # [bsz, ntokens]
            _, ids = torch.max(logit, 1, keepdim=True)

            if self.cfg.disc_s_in == 'embed':
                out = F.softmax(logit/1e-6, 1).unsqueeze(1)
                # [bsz, 1 vocab_size]

            # if eos token has already appeared, fill zeros
            ids, out, finished = self._pads_after_eos(ids, out, finished)

            # append generated token ids & outs at each step
            all_ids.append(ids)
            all_outs.append(out)
            all_logits.append(logit.unsqueeze(1))
            # for the next step
            embedding = self.embedding(ids)
            input_ = torch.cat([embedding, code.unsqueeze(1)], 2)

        # concatenate all the results
        max_ids = torch.cat(all_ids, 1).data.cpu().numpy() # [bsz, max_len]
        outs = torch.cat(all_outs, 1) # [bsz, max_len, vocab_size]
        self.logits = torch.cat(all_logits, 1) # [bsz, max_len]

        return max_ids, outs
