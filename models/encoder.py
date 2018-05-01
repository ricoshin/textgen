import logging

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.base_module import BaseModule
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence
from utils.utils import to_gpu

log = logging.getLogger('main')


class BaseEncoder(BaseModule):
    def __init__(self, cfg):
        super(BaseEncoder, self).__init__()
        self.cfg = cfg
        self.noise_radius = cfg.noise_radius
        self._is_add_noise = False

    def with_noise(self, *inputs):
        self._is_add_noise = True
        return self.__call__(*inputs)

    def forward(self, *inputs):
        code = self._encode(*inputs)
        # normalization
        # if self.cfg.code_norm:
        #     code = self._normalize(code)
        # unit gaussian noise
        if self._is_add_noise and self.noise_radius > 0:
            code = self._add_gaussian_noise_to(code)
            self._is_add_noise = False  # back to default
        return code

    def _add_gaussian_noise_to(self, code):
        # gaussian noise
        noise = torch.normal(means=torch.zeros(code.size()),
                             std=self.noise_radius)
        noise = to_gpu(self.cfg.cuda, Variable(noise))
        return code + noise

    def clip_grad_norm(self):
        nn.utils.clip_grad_norm(self.parameters(), self.cfg.clip)
        return self

    def decay_noise_radius(self):
        self.noise_radius = self.noise_radius * self.cfg.noise_anneal


class EncoderRNN(BaseEncoder):
    def __init__(self, cfg):
        super(EncoderRNN, self).__init__(cfg)

        self.encoder = nn.LSTM(input_size=cfg.embed_size_w,
                               hidden_size=cfg.hidden_size_w,
                               num_layers=cfg.nlayers,
                               dropout=cfg.dropout,
                               batch_first=True)
        #self.fc_layer = nn.Linear(cfg.hidden_size, 300)
        self._init_weights()

    def _encode(self, embed_in, lengths):
        # indices = [bsz, max_len], lengths = [bsz]

        # Embedding and pack
        packed_embeddings = pack_padded_sequence(input=embed_in,
                                                 lengths=lengths,
                                                 batch_first=True)
        # RNN encoder
        packed_output, state = self.encoder(packed_embeddings)
        hidden, cell = state  # last states (tuple the length of 2)
        code = hidden[-1]  # get hidden state of last layer of encoder

        #code = self.fc_layer(code)

        return code  # batch_size x hidden_size

    def _init_weights(self):
        # Unifrom initialization in the range of [-0.1, 0.1]
        initrange = 0.1

        # Initialize Encoder and Decoder Weights
        for p in self.encoder.parameters():
            p.data.uniform_(-initrange, initrange)


class EncoderCNN(BaseEncoder):
    def __init__(self, cfg):
        super(EncoderCNN, self).__init__(cfg)
        # Expected input dim
        #   : [bsz, c(embed or hidden size), h(1), w(max_len)]
        arch = cfg.arch_cnn

        # Convoutional layers
        self.convs = []
        for i in range(arch.n_conv):
            conv = nn.Conv2d(arch.c[i], arch.c[i + 1],
                             (1, arch.f[i]), arch.s[i])
            self.convs.append(conv)
            self.add_module("Conv(%d)" % (i + 1), conv)

        self.criterion_mse = nn.MSELoss()

    def _encode(self, embed_in, *therest):
        # NOTE : lengths can be used for pad masking
        if embed_in.size(1) < self.cfg.max_len:
            embed_in = self._append_zero_embeds(embed_in)
        elif embed_in.size(1) > self.cfg.max_len:
            embed_in = embed_in[:, :self.cfg.max_len, :]

        # [bsz, embed_size_w, 1, max_len]
        x = x_in = embed_in.permute(0, 2, 1).unsqueeze(2)

        for i, conv in enumerate(self.convs):
            if i < len(self.convs) - 1:
                x = F.leaky_relu(conv(x), 0.2)
            else:
                x = conv(x)

        code = x.squeeze(2).squeeze(2)  # batch size could be 1
        assert(len(code.size()) == 2)

        return code  # [bsz, hidden_size]

    def _append_zero_embeds(self, tensor):
        bsz, lengths, embed_size = tensor.size()
        pad_len = (self.cfg.max_len) - lengths
        if pad_len > 0:
            pads = torch.zeros([bsz, pad_len, embed_size])
            pads = Variable(pads, requires_grad=False).cuda()
            return torch.cat((tensor, pads), dim=1)
        else:
            return tensor


class VariationalRegularizer(BaseModule):
    def __init__(self, cfg):
        super(VariationalRegularizer, self).__init__()
        self.cfg = cfg
        self._sigma = None
        self._with_var = True

        self.mu_layers = torch.nn.Sequential(
            nn.Linear(cfg.hidden_size_w, cfg.hidden_size_w),
            nn.Tanh(),
            nn.Linear(cfg.hidden_size_w, cfg.hidden_size_w),
        )
        self.sigma_layers = torch.nn.Sequential(
            nn.Linear(cfg.hidden_size_w, cfg.hidden_size_w),
            nn.Tanh(),
            nn.Linear(cfg.hidden_size_w, cfg.hidden_size_w),
        )
        #self._init_weights()

    @property
    def sigma(self):
        if self._sigma is not None:
            return self._sigma.mean().data[0]
        else:
            return 0

    def with_var(self, enc_h):
        self._with_var = True
        return self(enc_h)

    def without_var(self, enc_h):
        self._with_var = False
        return self(enc_h)

    def forward(self, enc_h):
        #code = F.relu(code)
        mu = self.mu_layers(enc_h)
        # normalization
        if self.cfg.code_norm:
            mu = self._normalize(mu)

        if self._with_var:
            log_sigma = self.sigma_layers(enc_h)
            self._sigma = sigma = torch.exp(log_sigma)
            std = np.random.normal(0, 1, size=sigma.size())
            std = Variable(torch.from_numpy(std).float(), requires_grad=False)
            std = to_gpu(self.cfg.cuda, std)
            code = mu + sigma*std
        else:
            code = mu
            self._sigma = None
            self._with_var = True

        return code

    def clip_grad_norm(self):
        nn.utils.clip_grad_norm(self.parameters(), self.cfg.clip)
        return self

    def _init_weights(self):
        for name, param in self.named_parameters():
            if 'log_sigma' in name:
                param.data.normal_().mul_(0.0001)


class CodeSmoothingRegularizer(BaseModule):
    def __init__(self, cfg):
        super(CodeSmoothingRegularizer, self).__init__()
        self.cfg = cfg
        self.is_with_var = True  # default
        self.fc_logvar1 = nn.Linear(cfg.hidden_size_w, cfg.hidden_size_w)
        self.fc_logvar2 = nn.Linear(cfg.hidden_size_w, cfg.hidden_size_w)
        self._sigma = None
        #self._init_weights()

    @property
    def var(self):
        if self._sigma is not None:
            return self._sigma**2
        else:
            return 0

    def with_var(self, code):
        self.is_with_var = True  # default
        return self(code)

    def without_var(self, code):
        self.is_with_var = False
        return self(code)

    def with_directional_var(self, code, code_direction):
        code_dir = code_direction.ge(0)
        self.is_with_var = True
        return self(code, code_dir)

    def forward(self, code, code_dir=None):
        #code = F.relu(code)
        #code_new = self._reparameterize(code, logvar)
        if self.is_with_var:
            logvar = F.tanh(self.fc_logvar1(code))
            logvar = self.fc_logvar2(logvar)
            #import pdb; pdb.set_trace()
            self._sigma = sigma = logvar.mul(0.5).exp_() # always positive
            eps = Variable(sigma.data.new(sigma.size()).normal_())
            if code_dir is not None:
                eps_pos = eps.ge(0)
                eq = code_dir.eq(eps_pos)
                ne = eq^1
                eps = eps*eq.float() - eps*ne.float()
            noise = eps.detach().mul(sigma)
            #self._var = std.mean().data[0]
            #import pdb; pdb.set_trace()
            code = code + noise
        else:
            self._sigma = None

        self.is_with_var = True  # back to default

        # if self.cfg.code_norm:
        #      code = self._normalize(code)

        return code

    def _reparameterize(self, mu, logvar):
        if self.is_with_var:
            std = logvar.mul(0.5).exp_()
            eps = Variable(std.data.new(std.size()).normal_())
            var = eps.mul(std)
            self._var = std.mean().data[0]
            return var.add_(mu)
        else:
            self._var = None
            return mu

    def clip_grad_norm(self):
        nn.utils.clip_grad_norm(self.parameters(), self.cfg.clip)
        return self

    # def _init_weights(self):
    #     for p in self.fc_logvar.parameters():
    #         p.data.normal_().mul_(0.1)
