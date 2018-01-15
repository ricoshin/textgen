import logging
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import to_gpu

log = logging.getLogger('main')


class SampleDiscriminator(nn.Module):
    def __init__(self, max_len, filter_size, step_dim, embed_size, dropout):
        super(SampleDiscriminator, self).__init__()
        # Sentence should be represented as a matrix : [max_len x step_size]
        # Step represetation can be :
        #   - hidden states which each word is generated from
        #   - embeddings that were porduced from generated word indices
        # inputs.size() : [batch_size(N), 1(C), max_len(H), embed_size(W)]
        def next_height(in_size, filters_size, stride):
            return (in_size - filter_size) // stride + 1

        num_conv = 4
        strides = [1, 2, 2] # last_one should be calculated later
        filters = [3, 3, 3] # last_one should be calculated later
        channels = [step_dim] + [128*(2**(i)) for i in range(num_conv)]
        fc_layers = []

        heights = [max_len]
        for i in range(len(filters)):
            heights.append(next_height(heights[i], filters[i], strides[i]))

        # last layer (size dependant on the previous layer)
        filters += [heights[-1]]
        strides += [1]
        heights += [1]

        num_fc = 2
        size_fc  = [sum(channels[1:])] * (num_fc) + [1]

        log.debug(filters)  # filters = [3, 3, 3, 4]
        log.debug(strides)  # strides = [1, 2, 2, 1]
        log.debug(heights)  # heights = [21, 19, 9, 4, 1]
        log.debug(channels) # channels = [300, 128, 256, 512, 1024]
        log.debug(size_fc)  # size_fc = [1920, 1920, 1]

        ch = channels
        # main convoutional layers
        self.convs = nn.ModuleList(
            [nn.Conv2d(ch[i], ch[i+1], (filters[i], 1), strides[i])
                for i in range(num_conv)])

        ch = channels[1:]
        # wordwise attention layers
        self.word_attn1 = nn.ModuleList(
            [nn.Conv2d(ch[i], ch[i], (1, 1)) for i in range(num_conv-1)])
        self.word_attn2 = nn.ModuleList(
            [nn.Conv2d(ch[i], 1, (1, 1)) for i in range(num_conv-1)])

        # layerwise attention layers
        self.layer_attn1 = nn.ModuleList(
            [nn.Conv2d(ch[i], ch[i], (1, 1)) for i in range(num_conv)])
        self.layer_attn2 = nn.ModuleList(
            [nn.Conv2d(ch[i], 1, (1, 1)) for i in range(num_conv)])

        # fully-connected layers
        self.fc = nn.ModuleList(
            [nn.Linear(size_fc[i], size_fc[i+1]) for i in range(num_fc)])
        self.dropout = nn.Dropout(dropout)

        self.parameters


    def forward(self, x):
        attn_wordwise = []
        attn_layerwise = []
        attn_vectors = []

        # raw input : [batch_size, max_len, embed_size]
        x = x.unsqueeze(1).permute(0, 3, 2, 1)  # [bsz, embed_size, max_len, 1]

        for i in range(len(self.convs)): # when i = 0, channel[0] = 128
            x = F.relu(self.convs[i](x)) # [bsz, 128, 19, 1]

            if i < (len(self.convs)-1):
                # word-wise attention
                attn_w = F.tanh(self.word_attn1[i](x)) # [bsz, 128, 19, 1]
                attn_w = F.sigmoid(self.word_attn2[i](attn_w)) # [bsz, 1, 19, 1]
                attn_wordwise.append(attn_w.squeeze().data.cpu().numpy()) # [bsz, 19]
                x_a = torch.sum(x * attn_w.expand_as(x), 2, True) # [bsz, 128, 1, 1]
            else: # For the last layer (no wordwise attention)
                attn_wordwise.append(np.ones((x.size(0), 1), np.float32))
                x_a = x # x.size() : [bsz. 1024, 1, 1]

            # layer-wise attention
            attn_l = F.tanh(self.layer_attn1[i](x_a)) # [bsz, 128, 1, 1]
            attn_l = F.sigmoid(self.layer_attn2[i](attn_l)) # [bsz, 1, 1, 1]
            attn_layerwise.append(attn_l.squeeze().data.cpu().numpy()) # [bsz, 1]
            x_aa = x_a * attn_l.expand_as(x_a) # [bsz, 128, 1, 1]
            attn_vectors.append(x_aa.squeeze()) # [bsz, 128]

        x = self.dropout(torch.cat(attn_vectors, 1)) # [bsz, sum(channels)]

        # fully-connected layers
        for i, fc in enumerate(self.fc):
            x = fc(x) # [bsz, sum(channels)]
            if not i == len(self.fc) - 1:
                x = F.tanh(x)
        x = torch.mean(x) # [bsz], for WGAN loss
        return x, [attn_wordwise, attn_layerwise]

    def init_weights(self, layers):
        init_std = 0.02
        for layer in self.layers:
            try:
                layer.weight.data.normal_(0, init_std)
                layer.bias.data.fill_(0)
            except:
                pass

    @staticmethod
    def train_(cfg, disc, real_states, fake_states):
            # clamp parameters to a cube
        for p in disc.parameters():
            p.data.clamp_(-cfg.gan_clamp, cfg.gan_clamp) # [min,max] clamp
            # WGAN clamp (default:0.01)

        disc.train()
        disc.zero_grad()

        # loss / backprop
        err_d_real, attn_real, = disc(real_states.detach())
        one = to_gpu(cfg.cuda, torch.FloatTensor([1]))
        err_d_real.backward(one)

        # negative samples ----------------------------
        # loss / backprop
        err_d_fake, attn_fake, = disc(fake_states.detach())
        err_d_fake.backward(one * -1)

        err_d = -(err_d_real - err_d_fake)

        return ((err_d.data[0], err_d_real.data[0], err_d_fake.data[0]),
               (attn_real, attn_fake))
