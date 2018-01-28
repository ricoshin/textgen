import logging
import math

import torch
import torch.nn as nn
from torch.autograd import Variable

from train.train_helper import ResultPackage
from utils.utils import to_gpu

log = logging.getLogger('main')


class CodeDiscriminator(nn.Module):
    def __init__(self, cfg):
        super(CodeDiscriminator, self).__init__()
        # arguments default values
        #   ninput: args.nhidden=300
        #   noutput: 1
        #   layers: arch_d: 300-300

        # nhidden(in) --(layer1)-- 300 --(layer2)-- 300 --(layer3)-- 1(out)
        self.cfg = cfg
        ninput = cfg.hidden_size # 300(nhidden)
        noutput = 1

        activation = nn.LeakyReLU(0.2)
        layer_sizes = [ninput] + [int(x) for x in cfg.arch_d.split('-')]
        # [nhidden(300), 300, 300]
        self.layers = []

        # add_module here is required to define each layer with different name
        #    in the interative loop automatically according to desired architecture
        # By doing so, init & forward iterative codes can be made shorter
        for i in range(len(layer_sizes)-1):
            layer = nn.Linear(layer_sizes[i], layer_sizes[i+1])
            self.layers.append(layer)
            self.add_module("layer"+str(i+1), layer)

            # No batch normalization after first layer
            if i != 0:
                bn = nn.BatchNorm1d(layer_sizes[i+1], eps=1e-05, momentum=0.1)
                self.layers.append(bn)
                self.add_module("bn"+str(i+1), bn)

            self.layers.append(activation)
            self.add_module("activation"+str(i+1), activation)

        layer = nn.Linear(layer_sizes[-1], noutput)
        self.layers.append(layer)
        #self.add_module("layer"+str(len(self.layers)), layer)
        self.add_module("layer"+str(len(layer_sizes)), layer) # bug fix

        # gan_disc
        # MLP_D(
        #   (layer1): Linear(in_features=nhidden, out_features=300)
        #   (activation1): LeakyReLU(0.2)
        #   (layer2): Linear(in_features=300, out_features=300)
        #   (bn2): BatchNorm1d(300, eps=1e-05, momentum=0.1, affine=True)
        #   (activation2): LeakyReLU(0.2)
        #   (layer3): Linear(in_features=300, out_features=1)
        # )

        self._init_weights()

    def forward(self, x, train=False):
        self._check_train(train)

        for i, layer in enumerate(self.layers):
            x = layer(x)
        x = torch.mean(x)
        return x

    def _init_weights(self):
        init_std = 0.02
        for layer in self.layers:
            try:
                layer.weight.data.normal_(0, init_std)
                layer.bias.data.fill_(0)
            except:
                pass

    def _check_train(self, train):
        if train:
            self.train()
            self.zero_grad()
        else:
            self.eval()
