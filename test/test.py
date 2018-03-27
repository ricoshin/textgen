import logging
import os
import time
from collections import OrderedDict
from enum import Enum, auto, unique
from test.evaluate import evaluate_sents
from test.supervisor import TestingSupervisor

import numpy as np
import spacy
import torch
import torch.nn as nn
import torch.nn.functional as F
from loader.data import Batch
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

class Tester(object):
    def __init__(self, net):
        log.info("Testing start!")
        # set_random_seed(net.cfg)
        self.net = net
        self.cfg = net.cfg
        #self.fixed_noise = net.gen.make_noise_size_of(net.cfg.eval_size)

        self.test_sents = load_test_data(net.cfg)
        self.pos_one = to_gpu(net.cfg.cuda, torch.FloatTensor([1]))
        self.neg_one = self.pos_one * (-1)

        self.result = ResultWriter(net.cfg)
        self.sv = TestingSupervisor(net, self.result)
        #self.sv.interval_func_train.update({net.enc.decay_noise_radius: 200})

        self.num_sample = 10
        self.max_sample = 64
        spacy_en = spacy.load('en')
        self.tokenizer = lambda s: [tok.text for tok in spacy_en.tokenizer(s)]

        end_of_loop = False
        while not end_of_loop:
            end_of_loop = self.test_loop(self.cfg, self.net, self.sv)

    def test_loop(self, cfg, net, sv):
        """Main test loop"""

        modes = "\n\n" + " ".join([str(mode) for mode in TestMode])
        configs = ' [9]#samples(%d) [0]Exit: ' % self.num_sample
        try:
            input_ = int(input(modes + configs))

        except Exception as e:
            log.info('\n[!] EOF without reading any data! Try again.')
            return False

        if input_ == 0:
            return True

        elif input_ == 9:
            self.num_sample = \
                self._ask_how_many(self.num_sample, self.max_sample)
            return False

        elif input_ == TestMode.AUTOENCODE.value:
            log.info(TestMode.AUTOENCODE.start_msg)
            batch = self._ask_what_to_encode()
            decoded = self._autoencode_from_text(batch, 'fr')
            text = decoded.get_text_with_pair(batch.src, self.num_sample)

        elif input_ == TestMode.SAMPLE.value:
            log.info(TestMode.SAMPLE.start_msg)
            z = np.random.normal(0, 1, (cfg.batch_size, cfg.z_size))
            decoded = self._decode_from_z(z)
            text = decoded.get_text(self.num_sample)

        elif input_ == TestMode.INTERPOLATE.value:
            log.info(TestMode.INTERPOLATE.start_msg)
            z, dist = self._get_interpolated_z(self.num_sample)
            log.info('[!] Distance between 2 points: %f' % dist)
            decoded = self._decode_from_z(z)
            text = decoded.get_text(self.num_sample)

        else:
            log.warning('Unknown option!')
            return False

        log.info(text)
        return False

    def _ask_which_mode(self):
        mode = None
        modes = "\n\n" + " ".join([str(mode) for mode in TestMode])
        while mode is None:
            try:
                mode = input(modes + ' [0]Exit: ')
            except Exception as e:
                log.info('EOF without reading any data! Try again.')

    def _ask_how_many(self, num, max=None):
        if max is None:
            max = num
        out_num = None
        while out_num is None:
            msg = '[!] #Sample(ENTER to maintain %d)? : ' % num
            try:
                out_num = int(input(msg))
            except Exception as e:
                out_num = num

            if out_num > max:
                log.warning("#Sample can't be greater than %d" % max)
                out_num = None
        log.info("#Sample has been changed to %d!" % out_num)
        return out_num

    def _ask_what_to_encode(self):
        try:
            text = input('Input text! (Press ENTER for using dataset)\n>> ')
        except Exception as e:
            text = None

        if len(text) == 0:
            batch = self.net.data_ae.next()
        else:
            batch = self._make_batch_from_input(text)
        return batch

    def _make_batch_from_input(self, text):
        tokens = self.tokenizer(text)
        ids_src = self.net.vocab_w.words2ids(tokens)
        ids_tar = ids_src + [self.net.vocab_w.EOS_ID]
        num_unk = ids_src.count(self.net.vocab_w.UNK_ID)
        length = len(tokens)
        log.info('Packing one-sized batch.')
        log.info('Total : %d / Unknown : %d' % (length, num_unk))

        source = torch.LongTensor(np.array([ids_src]))
        target = torch.LongTensor(np.array([ids_tar])).view(-1)
        length = [length]
        return Batch(source, target, length).variable().cuda(self.cfg.cuda)

    def _get_interpolated_z(self, num_samples):
        # sample 2 points and compute the distance btwn them
        z_a = np.random.normal(0, 1, (1, self.cfg.z_size))
        z_b = np.random.normal(0, 1, (1, self.cfg.z_size))
        dist = np.sqrt(np.sum((z_a - z_b)**2))
        # get intermediate points by interpolation
        offset = (z_b - z_a) / num_samples
        z = np.vstack([z_a + offset * i for i in range(num_samples)])
        return z, dist

    def _autoencode_from_text(self, batch, decode_mode):
        self.net.set_modules_train_mode(False)
        # Build graph
        embed = self.net.embed_w(batch.src)
        code = self.net.enc(embed, batch.len)
        #code_var = self.net.reg.with_var(code)
        #cos_sim = F.cosine_similarity(code, code_var, dim=1).mean()
        if decode_mode == 'tf':
            decoded = self.net.dec.teacher_forcing(code, batch)
        elif decode_mode == 'fr':
            decoded = self.net.dec.free_running(code, max(batch.len))
        else:
            raise Exception("Unknown decode_mode type!")
        return decoded

    def _decode_from_z(self, z):
        self.net.set_modules_train_mode(True)
        # Build graph
        z = Variable(torch.FloatTensor(z))
        z = to_gpu(self.cfg.cuda, z)
        code_fake = self.net.gen(z)
        decoded = self.net.dec.free_running(code_fake, self.cfg.max_len)
        return decoded
