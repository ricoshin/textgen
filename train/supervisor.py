from contextlib import contextmanager
import json
import logging
import math
from os import path, listdir
from tqdm import tqdm

import torch

log = logging.getLogger('main')


class TrainingSupervisor(object):
    def __init__(self, net, result_writer):
        self.net = net
        self.cfg = net.cfg
        self.result = result_writer

        # NOTE : Fix later!
        interval_train = 500
        interval_eval = 500

        self.interval_func_train = {
            self._log_scalar_and_text: interval_train,
            self._save_scalar_and_text: interval_train,
            #self._save_data_and_module: interval_train,
            }

        self.interval_func_eval = {
            self._log_scalar_and_text: interval_eval,
            self._save_scalar_and_text: interval_eval,
            self._save_data_and_module: interval_eval,
            self._save_embedding: 1000,
            }

        self.interval_func_global = {
        }

        self._gan_schedule = self._init_gan_schedule()

        self.global_step = 0
        self.global_maxstep = math.ceil(  # a bit dirty
            (len(net.data_train)*net.cfg.epochs) / net.cfg.niter_ae)
        self.global_step_fname = path.join(self.cfg.log_dir, 'step.json')

        self._progress_bar = tqdm(initial=self.global_step,
                                  total=self.global_maxstep)

        self._load_snapshot_if_available()

    def __del__(self):
        self._progress_bar.close()

    @contextmanager
    def training_context(self):
        yield  # training procedure
        for function, step in self.interval_func_train.items():
            if self.global_step % step == 0: function()

    @contextmanager
    def evaluation_context(self):
        yield  # testing procedure
        for function, step in self.interval_func_eval.items():
            if self.global_step % step == 0: function()

    @property
    def niter_ae(self):
        return self.cfg.niter_ae  # constant

    @property
    def niter_gan(self):
        if self.net.data_ae.step.epoch >= len(self._gan_schedule):
            return 0  # for the last epoch
        else:
            return self._gan_schedule[self.net.data_ae.step.epoch]

    def is_end_of_training(self):
        for function, step in self.interval_func_global.items():
            if self.global_step % step == 0: function()
        self.global_step += 1
        self._update_progress_bar()
        return self.global_step > self.global_maxstep

    def is_evaluation(self):
        return any([self.global_step % step == 0
            for step in self.interval_func_eval.values()])

    def _update_progress_bar(self):
        diff = self.global_step - self._progress_bar.n
        self._progress_bar.update(diff)

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

    def _save_data_and_module(self):
        self._save_global_step()
        self.net.save_batch_schedulers()
        self.net.save_modules()
        #log.debug("Model saved.")

    def _load_snapshot_if_available(self):
        log_dir = listdir(self.cfg.log_dir)
        #import pdb; pdb.set_trace()
        if not any(fname.endswith(('.ckpt', '.pickle')) for fname in log_dir):
            log.info("Can't find model.ckpt files. "
                     "Training begins from the scratch!")
        else:
            self._load_global_step()
            self.net.load_batch_schedulers()
            self.net.load_modules()

    def _save_global_step(self):
        with open(self.global_step_fname, 'w') as f:
            dump_dict = dict(
                global_step=self.global_step,
                global_maxstep=self.global_maxstep,
                )
            json.dump(dump_dict, f, indent=4)

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

    def _init_gan_schedule(self):
        if self.cfg.niter_gan_schedule != "": # 2-4-6
            gan_schedule = self.cfg.niter_gan_schedule.split("-")
            gan_schedule =  [int(x) for x in gan_schedule]
            # for example: gan_schedule = [2, 4, 6]
        else:
            gan_schedule = []
        niter_gan = 1
        gan_schedule_list = []
        for i in range(self.cfg.epochs):
            if i in gan_schedule:
                niter_gan += 1
            gan_schedule_list.append(niter_gan)
        return gan_schedule_list
