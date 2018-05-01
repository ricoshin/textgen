import logging
import os
import time
from collections import OrderedDict
from enum import Enum, auto, unique
from random import randint
from test.evaluate import evaluate_sents
from test.supervisor import TestingSupervisor

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import spacy
import torch
import torch.nn as nn
import torch.nn.functional as F
from loader.data import Batch
from sklearn.manifold import TSNE
from torch.autograd import Variable
from train.train_helper import load_test_data, mask_output_target
from utils.utils import set_random_seed, to_gpu
from utils.writer import ResultWriter

log = logging.getLogger('main')
odict = OrderedDict


@unique
class TestMode(Enum):
    AUTOENCODE = auto()
    SAMPLE = auto()
    INTERPOLATE = auto()

    def __str__(self):
        return "[%d]%s" % (self.value, self.name.title())

    @property
    def start_msg(self):
        return self.name.title() + ' Mode!'


class Visualizer(object):
    def __init__(self, net):
        log.info("Testing start!")
        # set_random_seed(net.cfg)
        self.net = net
        self.cfg = net.cfg
        #self.fixed_noise = net.gen.make_noise_size_of(net.cfg.eval_size)

        self.result = ResultWriter(net.cfg)
        self.sv = TestingSupervisor(net, self.result)
        #self.sv.interval_func_train.update({net.enc.decay_noise_radius: 200})

        self.n_real = net.cfg.eval_size
        self.n_fake = net.cfg.eval_size
        self.n_noise = 20

        end_of_loop = False
        while not end_of_loop:
            end_of_loop = self.sample_loop(self.cfg, self.net, self.sv)

    def sample_loop(self, cfg, net, sv):
        """Main test loop"""
        # encode real
        code = list()
        batch = self.net.data_eval.next()
        log.info("Generating real codes..")
        code.append(self._generate_real_code(batch, False))
        for i in range(self.n_noise):
            log.info("Generating real codes with noise.. %d" % i)
            code.append(self._generate_real_code(batch, True))
        log.info("Generating fake codes...")
        code.append(self._generate_fake_code())
        log.info("Done!")

        # code_real : [eval_size, embed_size]
        # code_fake : [eval_size, embed_size]
        # code_real_var : [eval_size, embed_size] * num_sample

        # Visualize
        code = [c.data.cpu().numpy() for c in code]
        #import pdb; pdb.set_trace()
        tsne = TSNE(learning_rate=100, verbose=1)
        tsne_data = np.concatenate(code, axis=0)
        log.info("T-SNE transforming..")
        tsne_result = tsne.fit_transform(tsne_data)
        log.info("Done!")

        first = self.n_real
        second = self.n_real * (self.n_noise + 1)

        real_x = tsne_result[:first, 0]
        real_y = tsne_result[:first, 1]
        real_var_x = tsne_result[first:second, 0]
        real_var_y = tsne_result[first:second, 1]
        fake_x = tsne_result[second:, 0]
        fake_y = tsne_result[second:, 1]

        log.info("Plotting..")
        plt.clf()
        colors = ['C'+str(randint(0, 9)) for _ in range(self.n_real)]
        for i in range(self.n_real):
            plt.scatter(real_x[i], real_y[i], c=colors[i], s=2)
        for i in range(self.n_noise):
            x = real_var_x[i*self.n_real:(i+1)*self.n_real]
            y = real_var_y[i*self.n_real:(i+1)*self.n_real]
            for j in range(self.n_real):
                plt.scatter(x[j], y[j], c=colors[j], alpha=0.1, s=1)
        plt.scatter(fake_x, fake_y, c='k', s=2)
        log.info("Done!")

        plt.savefig('./%s_%s_%d.pdf' % (cfg.out_dir, cfg.name, sv.global_step),
                    format='pdf')

        return True

    def _generate_real_code(self, batch, with_var=False):
        self.net.set_modules_train_mode(True)

        embed = self.net.embed_w(batch.src)
        code = self.net.enc(embed, batch.len)
        if with_var:
            code = self.net.reg.with_var(code)
        return code

    def _generate_fake_code(self):
        self.net.set_modules_train_mode(True)
        code = self.net.gen.for_eval()
        return code
