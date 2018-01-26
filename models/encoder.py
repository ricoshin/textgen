import logging
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence

from nn.embedding import WordEmbedding
from utils.utils import to_gpu

log = logging.getLogger('main')


class Encoder(nn.Module):
    def __init__(self, cfg, vocab):
        super(Encoder, self).__init__()
        self.cfg = cfg
        self.noise_radius = cfg.noise_radius
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

    def _gen_gauss_noise(self, size):
        gauss_noise = torch.normal(means=torch.zeros(size),
                                   std=self.noise_radius)
        return to_gpu(self.cfg.cuda, Variable(gauss_noise))

    def _add_noise(self, ae_mode, code):
        if ae_mode and self.cfg.noise_radius > 0:
            code = code + self._gen_gauss_noise(code.size())
        return code

    def _store_grad_norm(self, grad):
        norm = torch.norm(grad, 2, 1)
        self.grad_norm = norm.detach().data.mean()
        return grad

    def _save_norm(self, ae_mode, code):
        if ae_mode and code.requires_grad:
            code.register_hook(self._store_grad_norm)

    def _normalize_code(self, code):
        norms = torch.norm(code, 2, 1)
        return torch.div(code, norms.unsqueeze(1).expand_as(code))

class EncoderRNN(Encoder):
    def __init__(self, cfg, vocab):
        super(EncoderRNN, self).__init__(cfg, vocab)

        # RNN Encoder
        self.encoder = nn.LSTM(input_size=cfg.embed_size,
                               hidden_size=cfg.hidden_size,
                               num_layers=cfg.nlayers,
                               dropout=cfg.dropout,
                               batch_first=True)
        self._init_weights()

    def _init_weights(self):
        # unifrom initialization in the range of [-0.1, 0.1]
        initrange = 0.1
        # Initialize Encoder and Decoder Weights
        for p in self.encoder.parameters():
            p.data.uniform_(-initrange, initrange)

    def forward(self, indices, lengths, ae_mode=False, train=False):
        # indices = [bsz, max_len], lengths = [bsz]
        assert(len(indices.size()) == 2)
        assert(len(lengths) == indices.size(0))
        self._check_train(train)

        # embedding and pack
        inputs = self.embedding(indices)
        packed_input = pack_padded_sequence(input=inputs,
                                            lengths=lengths,
                                            batch_first=True)
        # rnn encoder
        packed_output, states = self.encoder(packed_input)
        hidden, cell = states
        code = hidden[-1] # last hidden : [batch_size x hidden_size]

        # normalize hidden
        code = self._normalize_code(code)

        # for autoencdoer
        code = self._add_noise(ae_mode, code)
        self._save_norm(ae_mode, code)

        return code # the code!
