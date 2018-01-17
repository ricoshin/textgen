import logging
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

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
        self.input_size = input_size = self.get_input_size()

        def next_height(in_size, filter_size, stride):
            return (in_size - filter_size) // stride + 1

        num_conv = 4
        strides = [1, 2, 2] # last_one should be calculated later
        filters = [3, 3, 3] # last_one should be calculated later
        channels = [input_size] + [128*(2**(i)) for i in range(num_conv)]
        fc_layers = []

        heights = [cfg.max_len + 1] # including sos/eos
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
        self.dropout = nn.Dropout(cfg.dropout)


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

    def get_input_size(self):
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

    @staticmethod
    def train_(cfg, disc, real_in, fake_in):
        if cfg.disc_s_in == 'embed':
            # real_in.size() : [batch_size, max_len]
            # fake_in.size() : [batch_size*2, max_len, vocab_size]

            # normal embedding lookup
            real_in = disc.embed(real_in)

            # soft embedding lookup (3D x 2D matrix multiplication)
            max_len = fake_in.size(1)
            vocab_size = fake_in.size(2)
            embed_size = disc.embed.weight.size(1)

            fake_in = fake_in.view(-1, vocab_size)
            fake_in = torch.mm(fake_in, disc.embed.weight)
            fake_in = fake_in.view(-1, max_len, embed_size)
            # *_embed.size() : [batch_size, max_len, embed_size]

            # clamp parameters to a cube
        for p in disc.parameters():
            p.data.clamp_(-cfg.gan_clamp, cfg.gan_clamp) # [min,max] clamp
            # WGAN clamp (default:0.01)

        disc.train()
        disc.zero_grad()

        err_d_real, attn_real, = disc(real_in.detach())
        err_d_fake, attn_fake, = disc(fake_in.detach())

        one = to_gpu(cfg.cuda, torch.FloatTensor([1]))
        err_d_real.backward(one)
        err_d_fake.backward(one * -1)
        err_d = -(err_d_real - err_d_fake)

        return ((err_d.data[0], err_d_real.data[0], err_d_fake.data[0]),
               (attn_real, attn_fake))
