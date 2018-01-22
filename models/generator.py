import torch
import torch.nn as nn
from torch.autograd import Variable

from train.train_helper import ResultPackage
from utils.utils import to_gpu


class Generator(nn.Module):
    def __init__(self, ninput, noutput, layers,
                 activation=nn.ReLU(), gpu=False):
        super(Generator, self).__init__()
        # arguments default values
        #   ninput: args.z_size=100
        #   noutput: args.nhidden=300
        #   layers: arch_d: 300-300 

        # z_size(in) --(layer1)-- 300 --(layer2)-- 300 --(layer3)-- nhidden(out)
        self.ninput = ninput
        self.noutput = noutput
        self.gpu = gpu

        layer_sizes = [ninput] + [int(x) for x in layers.split('-')]
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

        self.init_weights()

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
        return x

    def init_weights(self):
        # Initialization with Gaussian distribution: N(0, 0.02)
        init_std = 0.02
        for layer in self.layers:
            try:
                layer.weight.data.normal_(0, init_std)
                layer.bias.data.fill_(0)
            except:
                pass

    def make_noise(self, cfg, num_samples=None):
        if num_samples is None:
            num_samples = cfg.batch_size
        noise = Variable(torch.ones(num_samples, cfg.z_size))
        noise = to_gpu(cfg.cuda, noise)
        noise.data.normal_(0, 1)
        return noise

    def generate(self, cfg, noise=None, train=True):
        if train:
            self.train()
            self.zero_grad()
        else:
            self.eval()
        if noise is None:
            noise = self.make_noise(cfg)
        return self(noise)
