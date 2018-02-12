import logging
import numpy as np

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

from nn.attention import MultiLinear4D, WordAttention, LayerAttention
from nn.embedding import WordEmbedding
from train.train_helper import ResultPackage
from utils.utils import Config, to_gpu

from models.encoder import Encoder

log = logging.getLogger('main')


class EncoderDiscModeWrapper(object):
    def __init__(self, enc_disc):
        self.enc_disc = enc_disc
        self.criterion_bce = self.enc_disc.criterion_bce

    def __call__(self, indices):
        return self.enc_disc(indices, lengths=None, noise=False,
                             save_grad_norm=False, disc_mode=True)

    def __str__(self):
        self_name = self.__class__.__name__
        enc_disc_name = self.enc_disc.__class__.__name__
        return self_name + " has the same architecture as " + enc_disc_name

    def parameters(self):
        return self.enc_disc.parameters()

    def cuda(self):
        self.enc_disc = self.enc_disc.cuda()
        return self

    def train(self):
        self.enc_disc.train()

    def eval(self):
        self.enc_disc.eval()

    def zero_grad(self):
        self.enc_disc.zero_grad()

    def state_dict(self):
        return self.enc_disc.state_dict()

    def load_state_dict(self, load):
        self.enc_disc.load_state_dict(load)


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
        self.attn = [int(x) for x in attn.split('-')]
        self.n_conv = len(self.c)

        assert(len(self.attn) == (self.n_conv -1)) # last attn does not exist
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

        self.n_mat = self.c[-1] # for dimension matching
        self.n_fc = 2
        self.fc = [self.n_mat] * (self.n_fc) + [1]

        self._log_debug([self.n_conv], "n_conv")
        self._log_debug(self.f, "filters")
        self._log_debug(self.s, "strides")
        self._log_debug(self.w, "widths")
        self._log_debug(self.c, "channels")
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


class EncoderDisc(Encoder):
    def __init__(self, cfg, embed):
        super(EncoderDisc, self).__init__(cfg, embed)
        # Sentence should be represented as a matrix : [max_len x step_size]
        # Step represetation can be :
        #   - hidden states which each word is generated from
        #   - embeddings that were porduced from generated word indices
        # inputs.size() : [batch_size(N), 1(C), max_len(H), word_embed_size(W)]
        arch = EncoderDiscArchitect(cfg)
        cfg.update(dict(arch_enc_disc=Config(arch.__dict__)))
        # expected input dim
        #   : [bsz, c(embed or hidden size), h(1), w(max_len)]
        # main convoutional layers
        self.convs_enc = []
        self.convs_enc_no_bias = [] # just for calculate zero pad
        for i in range(arch.n_conv):
            conv = nn.Conv2d(arch.c[i], arch.c[i+1], (1, arch.f[i]), arch.s[i])
            conv_nb = nn.Conv2d(1, 1, (1, arch.f[i]), arch.s[i], bias=False)
            self.convs_enc.append(conv)
            self.convs_enc_no_bias.append(conv_nb)
            self.add_module("MainConv(%d)" % (i+1), conv)
            self.add_module("MainConvNoBias(%d)" % (i+1), conv_nb)

        # last fc
        self.fc_enc = MultiLinear4D(arch.c[-1], cfg.hidden_size, dim=1,
                                    n_layers=1)

        if not cfg.with_attn:
            return

        self.convs_attn = []
        for i in range(arch.n_conv):
            conv = nn.Conv2d(arch.c[i], arch.c[i+1], (1, arch.f[i]), arch.s[i])
            self.convs_attn.append(conv)
            self.add_module("AttnConv(%d)" % (i+1), conv)

        c_ = arch.c[1:] # [300, 500, 700, 900]
        w_ = arch.w[1:] # [19, 9, 4, 1]
        #n_attns = [w // 4 for w in w_] # [4, 2, 1]


        # wordwise attention layers
        self.word_attns = []
        for i in range(arch.n_conv - 1):
            word_attn = WordAttention(c_[i], w_[i], arch.n_mat, arch.attn[i],
                                      cfg.word_temp, last_act='softmax')
            self.word_attns.append(word_attn)
            self.add_module("WordAttention(%d)" % (i+1), word_attn)

        # layerwise attention layer
        self.layer_attn = LayerAttention(arch.n_mat, arch.n_conv,
                                         cfg.layer_temp, last_act='softmax')

        # final fc layers for attention
        self.fc_attn = MultiLinear4D(arch.n_mat, 1, dim=1, n_layers=2)

        # binary CE
        self.criterion_bce = nn.BCELoss()


    def forward(self, indices, lengths, noise, save_grad_norm=False,
                disc_mode=False):
        # NOTE : lengths can be used for pad masking
        if indices.size(1) < self.cfg.max_len:
            indices = self._append_pads(indices)
        elif indices.size(1) > self.cfg.max_len:
            indices = indices[:, :self.cfg.max_len, :]

        x = self._adaptive_embedding(indices) # [bsz, max_len, word_embed_size]
        x = x_in = x.permute(0, 2, 1).unsqueeze(2) # [bsz, word_embed_size, 1, max_len]

        # generate mask for wordwise attention
        if disc_mode:
            x_enc = []
            pad_masks = self._generate_pad_masks(x)

        ###################
        # encoding layers #
        ###################
        for i, conv in enumerate(self.convs_enc):
            x = F.relu(conv(x))
            if disc_mode:
                x_enc.append(Variable(x.data, requires_grad=False))

        # last fc for encoding
        code = self.fc_enc(x).squeeze()

        # normalize code
        code = self._normalize_code(code)

        if noise and self.cfg.noise_radius > 0:
            code = self._add_noise(code)

        if save_grad_norm and code.requires_grad:
            code.register_hook(self._store_grad_norm)

        if (not disc_mode) or (not self.cfg.with_attn):
            assert(len(code.size()) == 2)
            return code # [bsz, hidden_size]

        ####################
        # attention layers #
        ####################
        w_ctx =[]
        w_attn = []
        x = x_in
        for i, conv in enumerate(self.convs_attn):
            # conv layers for attention
            x = F.relu(conv(x))
            # wordwise attention
            if i < (len(self.convs_attn)-1): # until 2nd last layer
                # compute wordwise attention
                ctx, attn = self.word_attns[i](x, x_enc[i], pad_masks[i])
                # ctx : [bsz, n_mat, 1, 1]
                # attn : [bsz, 1, 1, len]
                attn = attn.squeeze().data.cpu().numpy() # [bsz, max_len]
            else: # for the last layer (no wordwise attention)
                ctx = x # pass the x without attention
                attn = np.ones((x.size(0), 1), np.float32) # fill just one

            w_ctx.append(ctx)
            w_attn.append(attn)

        # stack along height dim
        try:
            w_ctx = torch.cat(w_ctx, dim=2) # [bsz, n_mat, n_layers, 1]
        except:
            import pdb; pdb.set_trace()
        # layerwise attention
        l_ctx, l_attn = self.layer_attn(w_ctx)
        # ctx : [bsz, n_mat, 1, 1]
        # attn : [bsz, 1, n_layers, 1]
        l_attn = l_attn.squeeze().permute(1,0).data.cpu().numpy()
        # [n_layers, bsz, 1]

        # final fc for attention
        x = self.fc_attn(l_ctx).squeeze() # [bsz]
        x = F.sigmoid(x)

        # w_attn : [n_layers, bsz, len]
        # layer_attn : [n_layers, bsz]
        return x, [w_attn, l_attn]

    def _append_pads(self, tensor):
        batch_size = tensor.size(0)
        pad_len = (self.cfg.max_len) - tensor.size(1)
        if pad_len > 0:
            pads = torch.ones([batch_size, pad_len]) * self.vocab.PAD_ID
            pads = Variable(pads, requires_grad=False).long().cuda()
            return torch.cat((tensor, pads), dim=1)
        else:
            return tensor

    def _adaptive_embedding(self, indices):
        if self.cfg.disc_s_in == 'embed':
            if len(indices.size()) == 2:
                # real case (indices) : [batch_size, max_len]
                return self.embed(indices, mode='hard')
            elif len(indices.size()) == 3:
                # fake case (onehots) : [batch_size, max_len, vocab_size]
                return self.embed(indices, mode='soft')
            else:
                raise Exception('Wrong embedding input dimension!')

    def _generate_pad_masks(self, x):
        # [bsz, word_embed_size, 1, max_len]
        x = Variable(x.data[:, 0].unsqueeze(1), requires_grad=False)
        # [bsz, 1, 1, max_len]
        masks = []
        for conv in self.convs_enc_no_bias:
            x = conv(x) # [bsz, 1, 1, max_len]
            zeros = x.eq(0).data.cpu()
            mask = torch.zeros(x.size()).masked_fill_(zeros, float('-inf'))
            masks.append(Variable(mask, requires_grad=False).cuda()) # mask pads as 0s
        return masks
