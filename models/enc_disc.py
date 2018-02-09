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
from models.architect import EncoderDiscArchitect

log = logging.getLogger('main')


class EncoderDiscModeWrapper(object):
    def __init__(self, enc_disc):
        self.enc_disc = enc_disc
        self.criterion_bce = self.enc_disc.criterion_bce

    def __call__(self, in_embed):
        return self.enc_disc(in_embed, lengths=None, noise=False,
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


class EncoderDisc(Encoder):
    def __init__(self, cfg):
        super(EncoderDisc, self).__init__(cfg)
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

        self.criterion_mse = nn.MSELoss()

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


    def forward(self, in_embed, noise, save_grad_norm=False, disc_mode=False):
        # NOTE : lengths can be used for pad masking
        if in_embed.size(1) < self.cfg.max_len:
            in_embed = self._append_zero_embeds(in_embed)
        elif in_embed.size(1) > self.cfg.max_len:
            in_embed = in_embed[:, :self.cfg.max_len, :]

        # [bsz, word_embed_size, 1, max_len]
        x = x_in = in_embed.permute(0, 2, 1).unsqueeze(2)

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

    def _append_zero_embeds(self, tensor):
        bsz, lengths, embed_size = tensor.size()
        pad_len = (self.cfg.max_len) - lengths
        if pad_len > 0:
            pads = torch.zeros([bsz, pad_len, embed_size])
            pads = Variable(pads, requires_grad=False).cuda()
            return torch.cat((tensor, pads), dim=1)
        else:
            return tensor

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
