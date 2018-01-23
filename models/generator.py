import torch
import torch.nn as nn
from torch.autograd import Variable

from train.train_helper import ResultPackage
from utils.utils import to_gpu


class Generator(nn.Module):
    def __init__(self, cfg):
        super(Generator, self).__init__()
        # arguments default values
        #   ninput: args.z_size=100
        #   noutput: args.nhidden=300
        #   layers: arch_d: 300-300

        # z_size(in) --(layer1)-- 300 --(layer2)-- 300 --(layer3)-- nhidden(out)
        self.cfg = cfg
        ninput = cfg.z_size
        noutput = cfg.hidden_size

        activation = nn.ReLU()
        layer_sizes = [ninput] + [int(x) for x in cfg.arch_g.split('-')]
        self.layers = []

        # add_module here is required to define each layer with different name
        #    in the interative loop automatically according to desired architecture
        # By doing so, init & forward iterative codes can be made shorter
        for i in range(len(layer_sizes)-1):
            layer = nn.Linear(layer_sizes[i], layer_sizes[i+1])
            self.layers.append(layer)
            self.add_module("layer"+str(i+1), layer)

            bn = nn.BatchNorm1d(layer_sizes[i+1], eps=1e-05, momentum=0.1)
            self.layers.append(bn)
            self.add_module("bn"+str(i+1), bn)

            self.layers.append(activation)
            self.add_module("activation"+str(i+1), activation)

        layer = nn.Linear(layer_sizes[-1], noutput)
        self.layers.append(layer)
        #self.add_module("layer"+str(len(self.layers)), layer)
        self.add_module("layer"+str(len(layer_sizes)), layer) # bug fix

        # Generator(
        #     (layer1): Linear(in_features=z_size, out_features=300)
        #     (bn1): BatchNorm1d(300, eps=1e-05, momentum=0.1, affine=True)
        #     (activation1): ReLU()
        #     (layer2): Linear(in_features=300, out_features=300)
        #     (bn2): BatchNorm1d(300, eps=1e-05, momentum=0.1, affine=True)
        #     (activation2): ReLU()
        #     (layer3): Linear(in_features=300, out_features=nhidden)
        # )

        self._init_weights()

    def forward(self, z=None, train=False):
        self._check_train(train)

        if z is None:
            x = self.make_noise()
        else:
            x = z

        for i, layer in enumerate(self.layers):
            x = layer(x)
        return x

    def _init_weights(self):
        # Initialization with Gaussian distribution: N(0, 0.02)
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

    def make_noise(self, num_samples=None):
        if num_samples is None:
            num_samples = self.cfg.batch_size
        noise = Variable(torch.ones(num_samples, self.cfg.z_size))
        noise = to_gpu(self.cfg.cuda, noise)
        noise.data.normal_(0, 1)
        return noise
