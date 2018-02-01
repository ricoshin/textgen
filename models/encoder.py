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
        self.vocab = vocab
        self.noise_radius = cfg.noise_radius
        self.grad_norm = None
        # word embedding
        self.embed = WordEmbedding(cfg, vocab.embed_mat)

    def forward(self, *input):
        raise NotImplementedError

    def _store_grad_norm(self, grad):
        norm = torch.norm(grad, 2, 1)
        self.grad_norm = norm.detach().data.mean()
        # use this when compute disc_c's gradient (register_hook)
        return grad

    def _normalize_code(self, code):
        norms = torch.norm(code, 2, 1)
        return torch.div(code, norms.unsqueeze(1).expand_as(code))

    def _add_noise(self, code):
        # gaussian noise
        noise = torch.normal(means=torch.zeros(code.size()),
                             std=self.noise_radius)
        noise = to_gpu(self.cfg.cuda, Variable(noise))
        return code + noise


class EncoderRNN(Encoder):
    def __init__(self, cfg, vocab):
        super(EncoderRNN, self).__init__(cfg, vocab)

        # RNN Encoder and Decoder
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

    def forward(self, indices, lengths, noise, save_grad_norm=False):
        code = self._encode(indices, lengths, noise)

        if save_grad_norm and code.requires_grad:
            code.register_hook(self._store_grad_norm)

        return code

    def _encode(self, indices, lengths, noise):
         # indices = [bsz, max_len], lengths = [bsz]
        assert(len(indices.size()) == 2)
        assert(len(lengths) == indices.size(0))

        # embedding and pack
        embeddings = self.embed(indices) # [bsz, max(lenghts), embed_dim]
        packed_embeddings = pack_padded_sequence(input=embeddings,
                                                 lengths=lengths,
                                                 batch_first=True)
        # rnn encoder
        packed_output, state = self.encoder(packed_embeddings)
        hidden, cell = state # last states (tuple the length of 2)
        code = hidden[-1]  # get hidden state of last layer of encoder

        # normalize code
        code = self._normalize_code(code)
        # add noise
        if noise and self.cfg.noise_radius > 0:
            code = self._add_noise(code)

        return code # batch_size x hidden_size
