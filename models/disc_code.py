import logging
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from models.base_module import BaseModule
from utils.writer import ResultWriter
from utils.utils import to_gpu

log = logging.getLogger('main')


class CodeDiscriminator(BaseModule):
    def __init__(self, cfg, code_size):
        super(CodeDiscriminator, self).__init__()
        # arguments default values
        #   ninput: args.nhidden=300
        #   noutput: 1
        #   layers: arch_d: 300-300

        # nhidden(in) --(layer1)-- 300 --(layer2)-- 300 --(layer3)-- 1(out)
        self.cfg = cfg
        ninput = code_size # 300(nhidden)
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

        self.wgan_linear = nn.Linear(layer_sizes[-1], noutput) # WGAN
        self.pred_linear = nn.Linear(layer_sizes[-1], noutput) # for prediction

        self.criterion_bce = nn.BCELoss()

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

    # override
    @property
    def tester(self, zero_grad=True):
        if zero_grad:
            self.zero_grad()
        return self.train(True) # turn on bath norm!

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            try:
                x = layer(x)
            except:
                import pdb; pdb.set_trace()
        x_wgan = torch.mean(self.wgan_linear(x))
        # x_pred = Variable(x.data, requires_grad=True)
        # x_pred = F.sigmoid(self.pred_linear(x_pred))
        return x_wgan

    def _init_weights(self):
        init_std = 0.02
        for layer in self.layers:
            try:
                layer.weight.data.normal_(0, init_std)
                layer.bias.data.fill_(0)
            except:
                pass

    def clamp_weights(self):
        # clamp parameters to a cube
        check_clamped = False
        from copy import deepcopy
        for name, param in self.named_parameters():
            if 'pred_linear' in name:
                continue
            #param_copy = deepcopy(param)
            param.data.clamp_(-self.cfg.gan_clamp, self.cfg.gan_clamp)
            # WGAN [min,max] clamp (default:0.01)
            #if not param_copy.equal(param):
            #    check_clamped = True
        #if not check_clamped:
        #    log.info('no clamping')
        return self
