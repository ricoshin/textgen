import logging
import numpy as np

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

from attention import MultiLinear4D, WordAttention, LayerAttention
from utils import to_gpu

log = logging.getLogger('main')


class SampleDiscriminator(nn.Module):
    def __init__(self, cfg, init_embed=None):
        super(SampleDiscriminator, self).__init__()
        # Sentence should be represented as a matrix : [max_len x step_size]
        # Step represetation can be :
        #   - hidden states which each word is generated from
        #   - embeddings that were porduced from generated word indices
        # inputs.size() : [batch_size(N), 1(C), max_len(H), embed_size(W)]
        if cfg.disc_s_in == 'embed':
            self.embed = nn.Embedding(cfg.vocab_size, cfg.embed_size)
            if (init_embed is not None) and cfg.load_glove:
                self.embed.weight.data.copy_(torch.from_numpy(init_embed))
            if cfg.load_glove and cfg.fix_embed:
                self.embed.weight.requires_grad = False

        self.cfg = cfg
        self.in_c = in_chann = self.get_in_c_size()

        def next_w(in_size, f_size, s_size):
            # in:width, f:filter, s:stride
            return (in_size - f_size) // s_size + 1

        n_conv = 4
        s = [1, 2, 2] # stride (last_one should be calculated later)
        f = [3, 3, 3] # filter (last_one should be calculated later)
        #c = [in_chann] + [128*(2**(i)) for i in range(n_conv)] # channel
        c = [in_chann] + [300, 400, 500, 600]


        w = [cfg.max_len + 1] # including sos/eos
        for i in range(len(f)):
            w.append(next_w(w[i], f[i], s[i]))

        # last layer (size dependant on the previous layer)
        f += [w[-1]]
        s += [1]
        w += [1]
        n_mat = c[-1] # for dimension matching
        n_fc = 2
        fc = [n_mat] * (n_fc) + [1]

        log.debug(f)  # filters = [3, 3, 3, 4]
        log.debug(s)  # strides = [1, 2, 2, 1]
        log.debug(w)  # widths = [21, 19, 9, 4, 1]
        log.debug(c) # channels = [300, 300, 500, 700, 900]
        log.debug(fc)  # size_fc = [1920, 1920, 1]

        # expected input dim
        #   : [bsz, c(embed or hidden size), h(1), w(max_len)]

        # main convoutional layers
        self.convs = []
        for i in range(n_conv):
            conv = nn.Conv2d(c[i], c[i+1], (1, f[i]), s[i], bias=True)
            self.convs.append(conv)
            self.add_module("MainConv(%d)" % (i+1), conv)
            #bias = nn.Parameter(torch.zeros([1, ch[i+1], heights[i+1], 1]))

        c_ = c[1:] # [300, 500, 700, 900]
        w_ = w[1:] # [19, 9, 4, 1]
        #n_attns = [w // 4 for w in w_] # [4, 2, 1]
        n_attns = [3, 1, 1]

        # wordwise attention layers
        self.word_attns = []
        for i in range(n_conv - 1):
            word_attn = WordAttention(c_[i], w_[i], n_mat, n_attns[i],
                                      last_act='softmax')
            self.word_attns.append(word_attn)
            self.add_module("WordAttention(%d)" % (i+1), word_attn)

        # layerwise attention layer
        self.layer_attn = LayerAttention(n_mat, n_conv, last_act='softmax')

        # final fc layers
        self.last_fc = MultiLinear4D(n_mat, 1, dim=1, n_layers=2)

        # dropout & binary CE
        self.dropout = nn.Dropout(cfg.dropout)
        self.criterion_bce = nn.BCELoss()


    def forward(self, x):
        # raw input : [batch_size, max_len, embed_size]
        x = x.permute(0, 2, 1).unsqueeze(2) # [bsz, embed_size, 1, max_len]

        # main conv & wordwise attention
        w_ctx =[]
        w_attn = []
        for i in range(len(self.convs)):
            # main conv
            x = F.relu(self.convs[i](x))
            # wordwise attention
            if i < (len(self.convs)-1): # before it reaches to the last layer
                # compute wordwise attention
                ctx, attn = self.word_attns[i](x)
                # ctx : [bsz, n_mat, 1, 1]
                # attn : [bsz, 1, 1, len]
                attn = attn.squeeze().data.cpu().numpy() # [bsz, max_len]
            else: # for the last layer (no wordwise attention)
                ctx = x # pass the x without attention
                attn = np.ones((x.size(0), 1), np.float32) # fill just one

            w_ctx.append(ctx)
            w_attn.append(attn)

        # stack along height dim
        x = torch.cat(w_ctx, dim=2) # [bsz, n_mat, n_layers, 1]

        # layerwise attention
        l_ctx, l_attn = self.layer_attn(x)
        # ctx : [bsz, n_mat, 1, 1]
        # attn : [bsz, 1, n_layers, 1]
        l_attn = l_attn.squeeze().permute(1,0).data.cpu().numpy()
        # [n_layers, bsz, 1]

        # final fc
        x = self.last_fc(l_ctx).squeeze() # [bsz]
        x = F.sigmoid(x)

        # w_attn : [n_layers, bsz, len]
        # layer_attn : [n_layers, bsz]
        return x, [w_attn, l_attn]

    def get_in_c_size(self):
        if self.cfg.disc_s_in == 'embed':
            return self.cfg.embed_size
        elif self.cfg.disc_s_in == 'hidden':
            return self.cfg.hidden_size
        else:
            raise Exception("Unknown disc input type!")

    def mask_sequence_with_n_inf(self, seqs, seq_lens):
        max_seq_len = seqs.size(1)
        masks = seqs.data.new(*seqs.size()).zero_()
        for mask, seq_len in zip(masks, seq_lens):
            seq_len = seq_len.data[0]
            if seq_len < max_seq_len:
                mask[seq_len:] = float('-inf')
        masks = Variable(masks, requires_grad=False)
        return seqs + masks

    def soft_embed(self, id_onehots):
        # id_onehot : [batch_size, max_len, vocab_size]
        # self.embed : [vocab_size, embed_size]
        max_len = id_onehots.size(1)
        vocab_size = id_onehots.size(2)
        embed_size = self.embed.weight.size(1)

        id_onehots = id_onehots.view(-1, vocab_size) # [bsz*max_len, vocab_size]
        embeddings = torch.mm(id_onehots, self.embed.weight)
        embeddings = embeddings.view(-1, max_len, embed_size)
        return embeddings

    @staticmethod
    def train_(cfg, disc, real_in, fake_in):
        disc.train()
        disc.zero_grad()

        if cfg.disc_s_in == 'embed':
            # real_in.size() : [batch_size, max_len]
            # fake_in.size() : [batch_size*2, max_len, vocab_size]
            # normal embedding lookup
            real_in = disc.embed(real_in)
            # soft embedding lookup (3D x 2D matrix multiplication)
            fake_in = disc.soft_embed(fake_in)
            # [batch_size, max_len, embed_size]

            # clamp parameters to a cube
        for p in disc.parameters():
            p.data.clamp_(-cfg.gan_clamp, cfg.gan_clamp) # [min,max] clamp
            # WGAN clamp (default:0.01)

        #real_in = real_in + real_in.eq(0).float() * (-1e-16)
        #fake_in = fake_in + fake_in.eq(0).float() * (-1e-16)

        pred_real, attn_real = disc(real_in.detach())
        pred_fake, attn_fake = disc(fake_in.detach())

        # loss
        label_real = to_gpu(cfg.cuda, Variable(torch.ones(pred_real.size())))
        label_fake = to_gpu(cfg.cuda, Variable(torch.zeros(pred_fake.size())))
        loss_real = disc.criterion_bce(pred_real, label_real)
        loss_fake = disc.criterion_bce(pred_fake, label_fake)
        loss_total = loss_real + loss_fake

        # accuracy
        real_mean = pred_real.mean()
        fake_mean = pred_fake.mean()

        # backprop.
        loss_real.backward()
        loss_fake.backward()

        return ((loss_total.data[0], loss_real.data[0], loss_fake.data[0]),
                (real_mean.data[0], fake_mean.data[0]),
                (attn_real, attn_fake))
