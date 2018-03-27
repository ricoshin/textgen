from contextlib import contextmanager
import json
import logging
import math
from os import path, listdir
from tqdm import tqdm

import torch

log = logging.getLogger('main')


class TestingSupervisor(object):
    def __init__(self, net, result_writer):
        self.net = net
        self.cfg = net.cfg
        self.result = result_writer

        self.global_step = 0
        self.global_maxstep = math.ceil(  # a bit dirty
            (len(net.data_train)*net.cfg.epochs) / net.cfg.niter_ae)
        self.global_step_fname = path.join(self.cfg.log_dir, 'step.json')

        self._load_snapshot()
        self._log_train_progress()

    def _log_scalar_and_text(self):
        self._log_train_progress()
        self.result.log_scalar_text()

    def _save_scalar_and_text(self):
        self.result.save_scalar(self.global_step)
        self.result.save_text(self.global_step)
        self.result.initialize_scalar_text()

    def _save_embedding(self):
        self.result.save_embedding(self.global_step)
        self.result.initialize_embedding()

    def _load_snapshot(self):
        log_dir = listdir(self.cfg.log_dir)
        #import pdb; pdb.set_trace()
        if not any(fname.endswith(('.ckpt', '.pickle')) for fname in log_dir):
            raise Exception("Can't find model.ckpt files in %s"
                            ""% self.cfg.log_dir)
        else:
            self._load_global_step()
            self.net.load_batch_schedulers()
            self.net.load_modules()

    def _load_global_step(self):
        with open(self.global_step_fname, 'r') as f:
            load_dict = json.load(f)
        log.info(load_dict)
        self.global_step = load_dict['global_step']
        self.global_maxstep = load_dict['global_maxstep']

    def _log_train_progress(self):
        #self._print_line()
        global_step = "Global step: %d/%d" % (
            self.global_step, self.global_maxstep)
        log.info("| %s | %s | %s |\n" % (
            self.cfg.name, self.net.data_ae.step, global_step))
