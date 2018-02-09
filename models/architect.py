import logging

log = logging.getLogger('main')


class CNNArchitect(object):
    def __init__(self, cfg):
        # when "0" strides or filters, they will be automatically computed
        # NOTE : add to parser.py later!
        s = "1-2-2-0"         # strides
        f = "3-3-3-0"         # filters
        c = "300-400-500-500"   # channels

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

        self._log_debug([self.n_conv], "n_conv")
        self._log_debug(self.f, "filters")
        self._log_debug(self.s, "strides")
        self._log_debug(self.w, "widths")
        self._log_debug(self.c, "channels")

    def _next_w(self, in_size, f_size, s_size):
        # in:width, f:filter, s:stride
        next_size = (in_size - f_size) // s_size + 1
        if next_size < 0:
            raise ValueError("feature map size can't be smaller than 0!")
        return next_size

    def _log_debug(self, int_list, name):
        str_list = ", ".join([str(i) for i in int_list])
        log.debug(name + ": [%s]" % str_list)


class EncoderDiscArchitect(object):
    def __init__(self, cfg):
        # when "0" strides or filters, they will be automatically computed
        # NOTE : add to parser.py later!
        s = "2-1-0"         # strides
        f = "5-5-0"         # filters
        c = "300-400-500"   # channels
        attn = "100-50"

        self.s = [int(x) for x in s.split('-') if x is not '0']
        self.f = [int(x) for x in f.split('-') if x is not '0']
        self.c = [int(x) for x in c.split('-')]

        self.n_conv = len(self.c)
        assert(len(self.s) == len(self.f)) # both must have the same len

        self.c = [self._get_in_c_size(cfg)] + self.c # input c
        self.w = [cfg.max_len]
        for i in range(len(self.f)):
            self.w.append(self._next_w(self.w[i], self.f[i], self.s[i]))

        if len(self.s) == (len(self.c) - 2):
            # last layer (size dependant on the previous layer)
            self.f += [self.w[-1]]
            self.s += [1]
            self.w += [1]

        self._log_debug([self.n_conv], "n_conv")
        self._log_debug(self.f, "filters")
        self._log_debug(self.s, "strides")
        self._log_debug(self.w, "widths")
        self._log_debug(self.c, "channels")

        if not cfg.with_attn:
            return

        self.attn = [int(x) for x in attn.split('-')]
        assert(len(self.attn) == (self.n_conv -1)) # last attn does not exist

        self.n_mat = self.c[-1] # for dimension matching
        self.n_fc = 2
        self.fc = [self.n_mat] * (self.n_fc) + [1]


        self._log_debug([self.n_mat], "matching-dim")
        self._log_debug(self.fc, "fully-connected")
        self._log_debug(self.attn, "attentions")

    def _get_in_c_size(self, cfg):
        if cfg.disc_s_in == 'embed':
            return cfg.word_embed_size
        elif cfg.disc_s_in == 'hidden':
            return cfg.hidden_size
        else:
            raise Exception("Unknown disc input type!")

    def _next_w(self, in_size, f_size, s_size):
        # in:width, f:filter, s:stride
        next_size = (in_size - f_size) // s_size + 1
        if next_size < 0:
            raise ValueError("feature map size can't be smaller than 0!")
        return next_size

    def _log_debug(self, int_list, name):
        str_list = ", ".join([str(i) for i in int_list])
        log.debug(name + ": [%s]" % str_list)
