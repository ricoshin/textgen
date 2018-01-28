import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

from nn.sparsemax import Sparsemax


class MultiLinear4D(nn.Module):
    def __init__(self, in_size, out_size, dim='', n_layers=2, bias=False,
                 activation=F.tanh, dropout=0.2):
        super(MultiLinear4D, self).__init__()
        # in_height.size() : [bsz, ch, 1, w]
        self.dim = dim
        self.out_size = out_size
        self.activation = activation
        ch = [in_size] * n_layers + [out_size]
        # multiple linear layers
        self.layers = nn.ModuleList(
            [nn.Linear(ch[i], ch[i+1], bias=bias) for i in range(n_layers)])
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # [bsz, ch, 1 , w] (when dim=1)
        all_dim = [dim for dim in x.size()]
        sel_dim = all_dim[self.dim] # ch
        all_dim[self.dim] = 1 # [bsz, 1, 1, w]
        rest_dim = 1
        for dim in all_dim:
            rest_dim *= dim # bsz*w
        x = x.view(rest_dim, sel_dim) # [bsz*w, ch]
        # stack layers
        for i, layer in enumerate(self.layers):
            if i < len(self.layers)-1:
                x = self.dropout(self.activation(layer(x)))
            else:
                x = self.dropout(layer(x)) # [bsz*w, 1]
        all_dim[self.dim] = self.out_size # [bsz, out_dim, 1, w]
        x = x.view(*all_dim) # [bsz, 1, 1, w]
        return x


class WordAttention(nn.Module):
    def __init__(self, cfg, in_chann, in_width, out_size, n_attns, temp=1.,
                 last_act='softmax'):
        super(WordAttention, self).__init__()
        # in_height.size() : [bsz, ch, 1, w]
        #assert(last_act in ['sigmoid', 'softmax', 'sparsemax'])
        self.temp = temp
        self.last_act = cfg.word_act
        self.attn_layers = []

        # attention layer
        for i in range(n_attns):
            attn_layer = MultiLinear4D(in_chann, 1, dim=1, n_layers=2)
            self.attn_layers.append(attn_layer)
            self.add_module("attn_layer(%d)" % (i+1), attn_layer)
        if last_act == 'sparsemax':
            self.sparsemax = Sparsemax(1, in_width)
        # compression layer (over multiple attention)
        # self.comp_layer = MultiLinear4D(in_width, 1, dim=3, n_layers=2)
        self.pool_layer = nn.MaxPool2d((1, in_width))
        # matching layer (for same dimension output)
        self.match_layer = MultiLinear4D(in_chann, out_size, dim=1, n_layers=1)

    def forward(self, x, mask):
        # x : [bsz, ch, 1, w]
        weights = []

        # multiple wordwise attention layers
        for attn_layer in self.attn_layers:
            score = attn_layer(x) # [bsz, 1, 1, w]
            if self.last_act == 'softmax':
                weight = F.softmax((score+mask)/self.temp, dim=3) # same
            elif self.last_act == 'sigmoid':
                weight = F.sigmoid(score+mask) # same
            elif self.last_act == 'sparsemax':
                weight = self.sparsemax(score)
            else:
                raise Exception('Unknown activation!')
            weights.append(weight)

        # sum all the attetion weights
        if len(weights) > 0:
            weight = torch.cat(weights, dim=1) # [bsz, len(weights), 1, w]
            weight = torch.sum(weight, dim=1, keepdim=True) # [bsz, 1, 1, w]
        else:
            weight = weights[0] # [bsz, 1, 1, w]

        # weighted feature
        x = x * weight.expand_as(x) # [bsz, ch, 1, w]
        # compression layer
        #x = self.comp_layer(x) # [bsz, ch, 1, 1]
        x = self.pool_layer(x) # [bsz, ch, 1, 1]
        # dimension matching layer
        x = self.match_layer(x) # [bsz, out_size, 1, 1]

        return x, weight

class LayerAttention(nn.Module):
    def __init__(self, in_chann, n_layers, temp=1., last_act='softmax'):
        super(LayerAttention, self).__init__()
        # in : [bsz, n_mat, n_layers, 1]
        assert(last_act in ['sigmoid', 'softmax', 'sparsemax'])
        self.temp = temp
        self.last_act = last_act
        self.attn_layer = MultiLinear4D(in_chann, 1, dim=1, n_layers=2)
        if last_act == 'sparsemax':
            self.sparsemax = Sparsemax(1, n_layers)

    def forward(self, x):
        # x : [bsz, n_mat, n_layers, 1]
        # layerwise attention layers
        score = self.attn_layer(x) # [bsz, 1, n_layers, 1]
        if self.last_act == 'softmax':
            weight = F.softmax(score/self.temp, dim=2) # same
        elif self.last_act == 'sigmoid':
            weight = F.sigmoid(score) # same
        elif self.last_act == 'sparsemax':
            weight = self.sparsemax(score)
        else:
            raise Exception('Unknown activation!')

        # weighted sum
        x = x * weight.expand_as(x) # [bsz, n_mat, n_layers, 1]
        x = torch.sum(x, dim=2, keepdim=True) # [bsz, n_mat, 1, 1]

        return x, weight.squeeze()
