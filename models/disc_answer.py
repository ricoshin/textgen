import logging
import math

import torch
import torch.nn as nn
from torch.autograd import Variable
from nn.embedding import WordEmbedding
from torch.nn.utils.rnn import pack_padded_sequence
import torch.nn.functional as F

from train.train_helper import ResultPackage
from utils.utils import to_gpu

log = logging.getLogger('main')


class AnswerDiscriminator(nn.Module):
    def __init__(self, cfg, vocab):
        super(AnswerDiscriminator, self).__init__()
        # arguments default values
        #   ninput: args.nhidden=300
        #   noutput: 1
        #   layers: arch_d: 300-300

        # nhidden(in) --(layer1)-- 300 --(layer2)-- 300 --(layer3)-- 1(out)
        self.cfg = cfg

        # the size of each embedding vector.
        self.embed_size = cfg.embed_size # D
        # size of the dictionary of embeddings. ques_vocab_len == ans_vocab_len.
        self.embed_num = cfg.vocab_size # V
        # label field vocab length. In this implementation: answer embedding dim.
        self.class_num = cfg.ans_embed_size # C
        class_i = 1 # Ci
        self.kernel_num = 100 # Co
        self.kernel_sizes = [3,4,5] # Ks
        self.embed = nn.Embedding(self.embed_num, self.embed_size)
        #self.embed = WordEmbedding(cfg, vocab.embed_mat)
        # self.convs1 = [nn.Conv2d(class_i, kernel_num, (K, embed_size)) for K in kernel_sizes]
        self.convs1 = nn.ModuleList([nn.Conv2d(
                    class_i, self.kernel_num, (K, self.embed_size)) for K in self.kernel_sizes])
        self.dropout = nn.Dropout(cfg.dropout)
        self.fc1 = nn.Linear(len(self.kernel_sizes)*self.kernel_num, self.class_num)

        # initialize weights
        init_std = 1/len(self.kernel_sizes)*self.kernel_num
        try:
            self.fc1.weight.data.normal_(0, init_std)
            self.fc1.bias.data.fill_(0)
        except:
            pass

    def conv_and_pool(self, x, conv):
        x = F.relu(conv(x)).squeeze(3)  # (N, Co, W)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        return x

    # x : decoder output
    def forward(self, x): # N : input length
        x = self.embed(x)  # (N, W, D)

        x = x.unsqueeze(1)  # (N, Ci, W, D)

        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs1]  # [(N, Co, W), ...]*len(Ks)

        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]  # [(N, Co), ...]*len(Ks)

        x = torch.cat(x, 1)

        x = self.dropout(x)  # (N, len(Ks)*Co)
        logit = self.fc1(x)  # (N, C)
        return logit
