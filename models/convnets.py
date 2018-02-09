import logging

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

from nn.embedding import WordEmbedding
from utils.utils import Config

from models.encoder import Encoder
from models.decoder import Decoder

log = logging.getLogger('main')


class CNNArchitect(object):
    def __init__(self, cfg):
        # when "0" strides or filters, they will be automatically computed
        # NOTE : add to parser.py later!
        s = "2-2-0"         # strides
        f = "5-5-0"         # filters
        c = "300-600-500"   # channels

        self.s = [int(x) for x in s.split('-') if x is not '0']
        self.f = [int(x) for x in f.split('-') if x is not '0']
        self.c = [int(x) for x in c.split('-')]

        self.n_conv = len(self.c)
        assert(len(self.s) == len(self.f)) # both must have the same len

        self.c = [cfg.word_embed_size] + self.c # input c
        self.w = [cfg.max_len]
        for i in range(len(self.f)):
            self.w.append(self._next_w(self.w[i], self.f[i], self.s[i]))

        if len(self.s) == (len(self.c) - 2):
            # last layer (size dependant on the previous layer)
            self.f += [self.w[-1]]
            self.s += [1]
            self.w += [1]

        self.w_r = [1]
        for i, j in zip(range(len(self.f)), reversed(range(len(self.f)))):
            self.w_r.append(self._next_w_r(self.w_r[i], self.f[j], self.s[j]))

        self._log_debug([self.n_conv], "n_conv")
        self._log_debug(self.f, "filters")
        self._log_debug(self.s, "strides")
        self._log_debug(self.w, "widths")
        self._log_debug(self.c, "channels")

    def _next_w(self, in_size, f_size, s_size):
        # in:width, f:filter, s:stride
        next_size = (in_size - f_size) / s_size + 1

        if not next_size.is_integer():
            raise ValueError("Feature map size has to be a whole number "
                             "so that it can be deconved symmetrically")
        if next_size < 0:
            raise ValueError("Feature map size can't be smaller than 0!")
        return int(next_size)

    def _next_w_r(self, in_size, f_size, s_size):
        next_size = (in_size - 1) * s_size + f_size
        return next_size

    def _log_debug(self, int_list, name):
        str_list = ", ".join([str(i) for i in int_list])
        log.debug(name + ": [%s]" % str_list)


class EncoderCNN(Encoder):
    def __init__(self, cfg, arch):
        super(EncoderCNN, self).__init__(cfg)
        # expected input dim
        #   : [bsz, c(embed or hidden size), h(1), w(max_len)]
        # convoutional layers
        self.arch = arch
        self.convs = []
        for i in range(arch.n_conv):
            conv = nn.Conv2d(arch.c[i], arch.c[i+1], (1, arch.f[i]), arch.s[i])
            self.convs.append(conv)
            self.add_module("Conv(%d)" % (i+1), conv)

        self.criterion_mse = nn.MSELoss()

    def forward(self, embed_in, noise=False, save_grad_norm=False):
        # NOTE : lengths can be used for pad masking
        if embed_in.size(1) < self.cfg.max_len:
            embed_in = self._append_zero_embeds(embed_in)
        elif embed_in.size(1) > self.cfg.max_len:
            embed_in = embed_in[:, :self.cfg.max_len, :]

        # [bsz, word_embed_size, 1, max_len]
        x = x_in = embed_in.permute(0, 2, 1).unsqueeze(2)

        for i, conv in enumerate(self.convs):
            x = F.relu(conv(x))

        # normalize code
        code = F.normalize(x.squeeze(), p=2, dim=1)

        if noise and self.cfg.noise_radius > 0:
            code = self._add_noise(code)

        if save_grad_norm and code.requires_grad:
            code.register_hook(self._store_grad_norm)

        assert(len(code.size()) == 2)
        return code # [bsz, hidden_size]

    def _append_zero_embeds(self, tensor):
        bsz, lengths, embed_size = tensor.size()
        pad_len = (self.cfg.max_len) - lengths
        if pad_len > 0:
            pads = torch.zeros([bsz, pad_len, embed_size])
            pads = Variable(pads, requires_grad=False).cuda()
            return torch.cat((tensor, pads), dim=1)
        else:
            return tensor


class DecoderCNN(Decoder):
    def __init__(self, cfg, arch):
        super(DecoderCNN, self).__init__(cfg)
        # expected input dim
        #   : [bsz, c(embed or hidden size), h(1), w(max_len)]
        # main convoutional layers
        self.arch = arch
        self.deconvs = []
        for i in reversed(range(arch.n_conv)):
            deconv = nn.ConvTranspose2d(arch.c[i+1], arch.c[i],
                                        (1, arch.f[i]), arch.s[i])
            self.deconvs.append(deconv)
            self.add_module("Deconv(%d)" % (i+1), deconv)

        self.criterion_ce = nn.CrossEntropyLoss()
        self.criterion_nll = nn.NLLLoss()

    def forward(self, code, save_grad_norm=False):
        # NOTE : lengths can be used for pad masking

        x = code.view(*code.size(), 1, 1)
        for i, deconv in enumerate(self.deconvs):
            x = deconv(x)

        embed = x.squeeze().permute(0, 2, 1)
        embed = F.normalize(embed, p=2, dim=2)

        if save_grad_norm and embed.requires_grad:
            embed.register_hook(self._store_grad_norm)

        assert(len(embed.size()) == 3)
        assert(embed.size(1) == self.cfg.max_len)
        assert(embed.size(2) == self.cfg.word_embed_size)

        return embed # [bsz, hidden_size]
