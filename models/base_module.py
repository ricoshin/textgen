import abc
import logging
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

log = logging.getLogger('main')


class BaseModule(nn.Module):
    def __init__(self):
        super(BaseModule, self).__init__()

    @property
    def trainer(self, zero_grad=True):
        if zero_grad:
            self.zero_grad()
        return self.train(True)

    @property
    def tester(self, zero_grad=True):
        if zero_grad:
            self.zero_grad()
        return self.train(False)

    def forward(self, *input):
        raise NotImplementedError

    def _normalize(self, code, p=2, dim=1):
        return F.normalize(code, p, dim)


class BaseAutoencoder(BaseModule):
    def __init__(self):
        super(BaseAutoencoder, self).__init__()
        self.eps = 1e-12
        self._grad_norm = None
        self.cfg = None # placeholder

    @property
    def grad_norm(self):
        if self._grad_norm is None:
            raise Exception("No saved gradient norm!")
        return self._grad_norm

    def forward(self, *input):
        raise NotImplementedError

    def save_ae_grad_norm_hook(self, grad):
        norm = torch.norm(grad, p=2, dim=1)
        self._grad_norm = norm.detach().data.mean() + self.eps
        return grad

    def scale_disc_grad_hook(self, grad):
        if isinstance(self, BaseEncoder):
            m_type = 'encoder'
            factor = -math.fabs(self.cfg.gan_to_enc)
        elif isinstance(self, BaseDecoder):
            m_type = 'decoder'
            factor = math.fabs(self.cfg.gan_to_dec)
        else:
            raise Exception("scale_disc_grad_hook is expected to be called by "
                            "BaseEncoder or BaseDecoder instance!")

        gan_norm = torch.norm(grad, p=2, dim=1)
        gan_norm = gan_norm.detach().data.mean() + self.eps

        if gan_norm == .0:
            log.warning("zero gan norm to %s!".format(m_type))
            normed_grad = grad
        else:
            normed_grad = grad * self.grad_norm / gan_norm

        normed_grad *= factor
        return normed_grad

    def _add_gaussian_noise_to(self, code):
        # gaussian noise
        noise = torch.normal(means=torch.zeros(code.size()),
                             std=self.noise_radius)
        noise = to_gpu(self.cfg.cuda, Variable(noise))
        return code + noise

class BaseEncoder(BaseAutoencoder):
    def __init__(self, cfg):
        super(BaseEncoder, self).__init__()
        self.cfg = cfg
        self.noise_radius = cfg.noise_radius
        self._is_add_noise = False

    def with_noise(self, *inputs):
        self._is_add_noise = True
        return self.__call__(*inputs)

    def forward(self, *inputs):
        code = self._encode(*inputs)

        # normalization
        code = self._normalize(code)

        # unit gaussian noise
        if self._is_add_noise and self.noise_radius > 0:
            code = self._add_gaussian_noise_to(code)
            self._is_add_noise = False # back to default

        return code


class BaseDecoder(BaseAutoencoder):
    def __init__(self):
        super(BaseDecoder, self).__init__()
