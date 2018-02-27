from enum import Enum
import logging

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

log = logging.getLogger('main')


class ConvnetType(Enum):
    ENCODER_ONLY = 0 # NOTE: redundant for now
    AUTOENCODER = 1
    ENC_DISC = 2

class ConvnetArchitect(object):
    """ You must set key attributes as names starting with 'arch_' prefix
       that will be finally removed when they're returned. """
    def __init__(self, cfg):
        self.cfg = cfg
        # when "0" strides or filters, they will be automatically computed
        # NOTE : add to parser.py later!
        self.s = "1-2-0"         # strides
        self.f = "5-5-0"         # filters
        self.c = "300-600-500"   # channels

    def design_model_of(self, convnet_type):
        self.convnet_type = convnet_type

        self._design_encoder()
        if convnet_type == ConvnetType.AUTOENCODER:
            self._design_decoder()
        elif convnet_type == ConvnetType.ENC_DISC:
            self._design_attention()

        return self._return_arch_attr_only()

    def _design_encoder(self):
        self.arch_s = [int(x) for x in self.s.split('-') if x is not '0']
        self.arch_f = [int(x) for x in self.f.split('-') if x is not '0']
        self.arch_c = [int(x) for x in self.c.split('-')]

        self.arch_n_conv = len(self.arch_c)
        assert(len(self.arch_s) == len(self.arch_f)) # both must have the same len

        self.arch_c = [self.cfg.word_embed_size] + self.arch_c # input c
        self.arch_w = [self.cfg.max_len]
        for i in range(len(self.arch_f)):
            self.arch_w.append(self._next_w(
                self.arch_w[i], self.arch_f[i], self.arch_s[i]))

        if len(self.arch_s) == (len(self.arch_c) - 2):
            # last layer (size dependant on the previous layer)
            self.arch_f += [self.arch_w[-1]]
            self.arch_s += [1]
            self.arch_w += [1]

        self._log_debug([self.arch_n_conv], "n_conv")
        self._log_debug(self.arch_f, "filters")
        self._log_debug(self.arch_s, "strides")
        self._log_debug(self.arch_w, "widths")
        self._log_debug(self.arch_c, "channels")

    def _design_decoder(self):
        if not 'f' in self.__dict__:
            raise Exception("Can't design decoder before "
                            "ConvnetArchitect._design_encoder call")
        self.arch_w_r = [1]
        for i, j in zip(range(len(self.arch_f)),
                        reversed(range(len(self.arch_f)))):
            self.w_r.append(self._next_w_r(
                self.arch_w_r[i], self.arch_f[j], self.arch_s[j]))

        self._log_debug(self.arch_w_r, "widths_reversed")

    def _design_attention(self):
        if not 'n_conv' in self.__dict__:
            raise Exception("Can't design attention before "
                            "ConvnetArchitect._design_encoder call")
        self.arch_attn = [int(x) for x in attn.split('-')]
        # last attn does not exist
        assert(len(self.arch_attn) == (self.arch_n_conv -1))

        self.arch_n_mat = self.arch_c[-1] # for dimension matching
        self.arch_n_fc = 2
        self.arch_fc = [self.arch_n_mat] * (self.arch_n_fc) + [1]

        self._log_debug([self.arch_n_mat], "matching-dim")
        self._log_debug(self.arch_fc, "fully-connected")
        self._log_debug(self.arch_attn, "attentions")

    def _return_arch_attr_only(self):
        arch_attr = dict()
        for name, value in self.__dict__.items():
            if 'arch_' in name:
                arch_attr.update({name[5:]: value})

        return arch_attr

    def _next_w(self, in_size, f_size, s_size):
        # in:width, f:filter, s:stride
        next_size = (in_size - f_size) / s_size + 1

        if (self.convnet_type is ConvnetType.AUTOENCODER and
            not next_size.is_integer()):
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
