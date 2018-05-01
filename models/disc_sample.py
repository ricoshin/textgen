import logging

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.base_module import BaseModule
from models.disc_code import CodeDiscriminator
from models.encoder import EncoderCNN
from nn.attention import LayerAttention, MultiLinear4D, WordAttention
from torch.autograd import Variable
from utils.utils import to_gpu
from utils.writer import ResultWriter

log = logging.getLogger('main')


class SampleDiscriminator(BaseModule):
    def __init__(self, cfg, input_size):
        super(SampleDiscriminator, self).__init__()
        # Sentence should be represented as a matrix : [max_len x step_size]
        # Step represetation can be :
        #   - hidden states which each word is generated from
        #   - embeddings that were porduced from generated word indices
        # inputs.size() : [batch_size(N), 1(C), max_len(H), embed_size_w(W)]
        self.cfg = cfg
        self.enc = EncoderCNN(cfg)
        self.disc = CodeDiscriminator(cfg, input_size)

    def forward(self, embed, code):
        code_new = self.enc(embed)
        code_all = torch.cat([code_new, code], dim=1)
        disc_out = CodeDiscriminator(code_all)
        return disc_out
