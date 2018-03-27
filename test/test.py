import logging
import os
import time
from collections import OrderedDict
from test.evaluate import evaluate_sents

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from test.supervisor import TrainingSupervisor
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
        return "[%d] %s" % (self.value, self.name.title())

class Tester(object):
    def __init__(self, net):
        log.info("Training start!")
        # set_random_seed(net.cfg)
        self.net = net
        self.cfg = net.cfg
        #self.fixed_noise = net.gen.make_noise_size_of(net.cfg.eval_size)

        self.test_sents = load_test_data(net.cfg)
        self.pos_one = to_gpu(net.cfg.cuda, torch.FloatTensor([1]))
        self.neg_one = self.pos_one * (-1)

        self.result = ResultWriter(net.cfg)
        self.sv = TrainingSupervisor(net, self.result)
        #self.sv.interval_func_train.update({net.enc.decay_noise_radius: 200})

        end_of_loop = False
        while not end_of_loop:
            end_of_loop = self.test_loop(self.cfg, self.net, self.sv)

    def test_loop(self, cfg, net, sv):
        """Main test loop"""
        try:
            choice = "\n\n" + " ".join([mode for mode in TestMode])
            input_ = input(choice +  '[0] Exit: ')

        except Exception as e:
            log.info('EOF without reading any data! Try again.')
            return False

        if input_ == 0: return True

        if input_ == TestMode.AUTOENCODE.value:
            batch = net.data_ae.next()
            self._eval_autoencoder

        elif input_ == TestMode.SAMPLE.value:
            log.info('[Inference] Random sampling mode')
            z = np.random.normal(0, 1, (cfg.batch_size, cfg.z_dim))
            text = self._generate_text_from_z(z)

        elif input_ == TestMode.INTERPOLATE.value
            log.info('[Inference] Latent space walking mode')
            z, dist = self._get_interpolated_z(20)
            log.info('[Inference] Distance between 2 points: %f' % dist)
            text = self._generate_text_from_z(z)

        return False

    def _get_interpolated_z(self, num_samples):
        # sample 2 points and compute the distance btwn them
        z_a = np.random.normal(0, 1, (1, self.cfg.z_dim))
        z_b = np.random.normal(0, 1, (1, self.cfg.z_dim))
        dist = np.sqrt(np.sum((z_a - z_b)**2))
        # get intermediate points by interpolation
        offset = (z_b - z_a) / num_samples
        z = np.vstack([z_a + offset*i for i in range(num_samples)])
        z = Variable(torch.FloatTensor(z))
        return z, dist

    def _generate_text_from_z(self, z, name="Generated"):
        self.net.set_modules_train_mode(True)

        # Build graph
        code_fake = self.net.gen(z)
        decoded = self.net.dec.free_running(code_fake, self.cfg.max_len)

        code_fake_embed = ResultWriter.Embedding(
            embed=code_fake.data, text=decoded.get_text_batch())

        self.result.add(name, odict(
            code=code_fake_embed,
            text=decoded.get_text(),
        ))

    def _eval_autoencoder(self, batch, decode_mode, name='AE_eval'):
        name += ('/' + decode_mode)
        self.net.set_modules_train_mode(False)

        # Build graph
        embed = self.net.embed_w(batch.src)
        code_t, code_w = self.net.enc(embed, batch.len)
        #code_var = self.net.reg.with_var(code)
        #cos_sim = F.cosine_similarity(code, code_var, dim=1).mean()

        if decode_mode == 'tf':
            tags, words = self.net.dec.teacher_forcing(code_t, code_w, batch)
        elif decode_mode == 'fr':
            tags, words = self.net.dec.free_running(
                code_t, code_w, max(batch.len))
        else:
            raise Exception("Unknown decode_mode type!")

        # Compute word prediction loss and accuracy
        loss_t, acc_t = self._compute_recontruction_loss_acc(
            tags.prob, batch.tar_tag, len(self.net.vocab_t))
        loss_w, acc_w = self._compute_recontruction_loss_acc(
            words.prob, batch.tar, len(self.net.vocab_w))

        code_t_embed = ResultWriter.Embedding(
            embed=code_t.data,
            text=tags.get_text_batch(),
            tag='embed_tags')
        code_w_embed = ResultWriter.Embedding(
            embed=code_w.data,
            text=words.get_text_batch(),
            tag='embed_words')

        self.result.add(name, odict(
            loss_tag=loss_t.data[0],
            loss_word=loss_w.data[0],
            acc_tag=acc_t.data[0],
            acc_word=acc_w.data[0],
            embed_tag=code_t_embed,
            embed_word=code_w_embed,
            #cosim=cos_sim.data[0],
            #var=self.net.reg.var,
            noise=self.net.enc.noise_radius,
            txt_tags=tags.get_text_with_pair(batch.src_tag),
            txt_words=words.get_text_with_pair(batch.src),
        ))
