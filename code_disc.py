import logging
import math

import torch
import torch.nn as nn
from torch.autograd import Variable

from utils import to_gpu

log = logging.getLogger('main')


class CodeDiscriminator(nn.Module):
    def __init__(self, ninput, noutput, layers,
                 activation=nn.LeakyReLU(0.2), gpu=False):
        super(CodeDiscriminator, self).__init__()
        # arguments default values
        #   ninput: args.nhidden=300
        #   noutput: 1
        #   layers: arch_d: 300-300

        # nhidden(in) --(layer1)-- 300 --(layer2)-- 300 --(layer3)-- 1(out)
        self.ninput = ninput # 300(nhidden)
        self.noutput = noutput # 1
        self.gpu = gpu

        layer_sizes = [ninput] + [int(x) for x in layers.split('-')]
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

        self.init_weights()

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
        x = torch.mean(x)
        return x

    def init_weights(self):
        init_std = 0.02
        for layer in self.layers:
            try:
                layer.weight.data.normal_(0, init_std)
                layer.bias.data.fill_(0)
            except:
                pass

    @staticmethod
    def train_(cfg, disc, ae, real_hidden, fake_hidden):
            # clamp parameters to a cube
        for p in disc.parameters():
            p.data.clamp_(-cfg.gan_clamp, cfg.gan_clamp) # [min,max] clamp
            # WGAN clamp (default:0.01)

        disc.train()
        disc.zero_grad()

        # positive samples ----------------------------
        def grad_hook(grad):
            # Gradient norm: regularize to be same
            # code_grad_gan * code_grad_ae / norm(code_grad_gan)

            # regularize GAN gradient in AE(encoder only) gradient scale
            # GAN gradient * [norm(Encoder gradient) / norm(GAN gradient)]
            if cfg.ae_grad_norm: # default:True / norm code gradient from critic->encoder
                gan_norm = torch.norm(grad, 2, 1).detach().data.mean()
                if gan_norm == .0:
                    log.warning("zero code_gan norm!")
                    normed_grad = grad
                else:
                    normed_grad = grad * ae.enc_grad_norm / gan_norm
                # grad : gradient from GAN
                # aeoder.grad_norm : norm(gradient from AE)
                # gan_norm : norm(gradient from GAN)
            else:
                normed_grad = grad

            # weight factor and sign flip
            normed_grad *= -math.fabs(cfg.gan_toenc)
            # math.fabs() : same as abs() but converts its argument to float if it can
            #   (otherwise, throws an exception)
            # args.gan_toenc: weight factor passing gradient from gan to encoder
            #   default: -0.01
            return normed_grad # -0.01 <- why flipping?

        real_hidden.register_hook(grad_hook) # normed_grad
        # loss / backprop
        err_d_real = disc(real_hidden)
        one = to_gpu(cfg.cuda, torch.FloatTensor([1]))
        err_d_real.backward(one)

        # negative samples ----------------------------
        # loss / backprop
        err_d_fake = disc(fake_hidden.detach())
        err_d_fake.backward(one * -1)

        # `clip_grad_norm` to prvent exploding gradient problem in RNNs / LSTMs
        torch.nn.utils.clip_grad_norm(ae.parameters(), cfg.clip)

        err_d = -(err_d_real - err_d_fake)

        return err_d.data[0], err_d_real.data[0], err_d_fake.data[0]
