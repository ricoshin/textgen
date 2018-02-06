import logging
import numpy as np

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

from nn.attention import MultiLinear4D, WordAttention, LayerAttention
from nn.embedding import WordEmbedding
from train.train_helper import ResultPackage
from utils.utils import to_gpu

log = logging.getLogger('main')


class SampleDiscriminator(nn.Module):
    def __init__(self, cfg, vocab):
        super(SampleDiscriminator, self).__init__()
        # Sentence should be represented as a matrix : [max_len x step_size]
        # Step represetation can be :
        #   - hidden states which each word is generated from
        #   - embeddings that were porduced from generated word indices
        # inputs.size() : [batch_size(N), 1(C), max_len(H), embed_size(W)]
        self.cfg = cfg
        self.in_c = in_chann = self._get_in_c_size()
        if cfg.disc_s_in == 'embed':
            self.embedding = WordEmbedding(cfg, vocab.embed_mat)

        def next_w(in_size, f_size, s_size):
            # in:width, f:filter, s:stride
            return (in_size - f_size) // s_size + 1

        n_conv = 4
        s = [1, 2, 2] # stride (last_one should be calculated later)
        f = [3, 3, 3] # filter (last_one should be calculated later)
        #c = [in_chann] + [128*(2**(i)) for i in range(n_conv)] # channel
        c = [in_chann] + [300, 500, 700, 900]


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
        import pdb; pdb.set_trace()
        # expected input dim
        #   : [bsz, c(embed or hidden size), h(1), w(max_len)]

        # main convoutional layers
        self.convs = []
        self.convs_no_bias = [] # just for calculate zero pad
        for i in range(n_conv):
            conv = nn.Conv2d(c[i], c[i+1], (1, f[i]), s[i], bias=True)
            conv_nb = nn.Conv2d(1, 1, (1, f[i]), s[i], bias=False)
            self.convs.append(conv)
            self.convs_no_bias.append(conv_nb)
            self.add_module("MainConv(%d)" % (i+1), conv)
            self.add_module("MainConvNoBias(%d)" % (i+1), conv_nb)
            #bias = nn.Parameter(torch.zeros([1, ch[i+1], heights[i+1], 1]))

        c_ = c[1:] # [300, 500, 700, 900]
        w_ = w[1:] # [19, 9, 4, 1]
        #n_attns = [w // 4 for w in w_] # [4, 2, 1]
        n_attns = [3, 2, 1]

        # wordwise attention layers
        self.word_attns = []
        for i in range(n_conv - 1):
            word_attn = WordAttention(cfg, c_[i], w_[i], n_mat, n_attns[i],
                                      cfg.word_temp, last_act='softmax')
            self.word_attns.append(word_attn)
            self.add_module("WordAttention(%d)" % (i+1), word_attn)

        # layerwise attention layer
        self.layer_attn = LayerAttention(n_mat, n_conv, cfg.layer_temp,
                                         last_act='softmax')

        # final fc layers for attention
        self.last_fc_attn = MultiLinear4D(n_mat, 1, dim=1, n_layers=2)

        # final fc layers for reconstruction
        code_dim = cfg.hidden_size
        self.last_fc_recon = MultiLinear4D(c[-1], code_dim, dim=1, n_layers=2)

        # dropout & binary CE
        self.dropout = nn.Dropout(cfg.dropout)
        self.criterion_bce = nn.BCELoss()
        #self.criterion_cs = F.cosine_similarity()

    def forward(self, x, train=False):
        self._check_train(train)
        x = self._adaptive_embedding(x) # [bsz, max_len, embed_size]
        x = x.permute(0, 2, 1).unsqueeze(2) # [bsz, embed_size, 1, max_len]

        # generate mask for wordwise attention
        pad_masks = self._generate_pad_masks(x)

        # main conv & wordwise attention
        w_ctx =[]
        w_attn = []
        for i in range(len(self.convs)):
            # main conv
            x = F.relu(self.convs[i](x))
            # wordwise attention
            if i < (len(self.convs)-1): # before it reaches to the last layer
                # compute wordwise attention
                ctx, attn = self.word_attns[i](x, pad_masks[i])
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
            x_a = torch.cat(w_ctx, dim=2) # [bsz, n_mat, n_layers, 1]
        except:
            import pdb; pdb.set_trace()
        # layerwise attention
        l_ctx, l_attn = self.layer_attn(x_a)
        # ctx : [bsz, n_mat, 1, 1]
        # attn : [bsz, 1, n_layers, 1]
        l_attn = l_attn.squeeze().permute(1,0).data.cpu().numpy()
        # [n_layers, bsz, 1]

        # final fc for attention
        x_a = self.last_fc_attn(l_ctx).squeeze() # [bsz]
        x_a = F.sigmoid(x_a)

        # w_attn : [n_layers, bsz, len]
        # layer_attn : [n_layers, bsz]
        return x_a, [w_attn, l_attn]

    def _check_train(self, train):
        if train:
            self.train()
            self.zero_grad()
        else:
            self.eval()

    def _adaptive_embedding(self, indices):
        if self.cfg.disc_s_in == 'embed':
            if len(indices.size()) == 2:
                # real case (indices) : [batch_size, max_len]
                return self.embedding(indices, mode='hard')
            elif len(indices.size()) == 3:
                # fake case (onehots) : [batch_size, max_len, vocab_size]
                return self.embedding(indices, mode='soft')
            else:
                raise Exception('Wrong embedding input dimension!')

    def _get_in_c_size(self):
        if self.cfg.disc_s_in == 'embed':
            return self.cfg.embed_size
        elif self.cfg.disc_s_in == 'hidden':
            return self.cfg.hidden_size
        else:
            raise Exception("Unknown disc input type!")

    def _generate_pad_masks(self, x):
        # [bsz, embed_size, 1, max_len]
        x = Variable(x.data[:, 0].unsqueeze(1), requires_grad=False)
        # [bsz, 1, 1, max_len]
        masks = []
        for conv in self.convs_no_bias:
            x = conv(x) # [bsz, 1, 1, max_len]
            zeros = x.eq(0).data.cpu()
            mask = torch.zeros(x.size()).masked_fill_(zeros, float('-inf'))
            masks.append(Variable(mask, requires_grad=False).cuda()) # mask pads as 0s
        return masks

    # NOTE : redundant code. remove later
    def mask_sequence_with_n_inf(self, seqs, seq_lens):
        max_seq_len = seqs.size(1)
        masks = seqs.data.new(*seqs.size()).zero_()
        for mask, seq_len in zip(masks, seq_lens):
            seq_len = seq_len.data[0]
            if seq_len < max_seq_len:
                mask[seq_len:] = float('-inf')
        masks = Variable(masks, requires_grad=False)
        return seqs + masks
