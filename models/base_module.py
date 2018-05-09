import abc
import logging
import math

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

from utils.utils import to_gpu

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

    def clip_grad_norm_(self):
        nn.utils.clip_grad_norm_(self.parameters(), self.cfg.clip)
        return self

    def forward(self, *input):
        raise NotImplementedError

    def _normalize(self, code, p=2, dim=1):
        return F.normalize(code, p, dim)
