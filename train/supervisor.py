import json
import logging
from os import path, listdir
from tqdm import tqdm

import torch

from loader.corpus import BatchIterator

log = logging.getLogger('main')

class TrainingSupervisor(object):
    def __init__(self, net):
        self.net = net
        self.cfg = net.cfg

        # all steps should start from 0
        self.epoch_step = 0
        self.global_step = 0
        self.epoch_total = self.cfg.epochs

        self.niters_ae = self.cfg.niters_ae
        self.gan_schedule = self._init_gan_schedule()
        self.niters_gan = self.gan_schedule[0]
        self.step_fname = path.join(self.cfg.log_dir, 'schedule.json')

        self.load_if_exists()

    @property
    def ref_batch_iterator(self):
        if not hasattr(self, '_ref_batch_iterator'):
            raise Exception("You must call TrainingSupervisor."
                            "register_ref_batch_iterator() first!")
        return self._ref_batch_iterator

    def register_ref_batch_iterator(self, batch_iterator):
        if not isinstance(batch_iterator, BatchIterator):
            raise Exception("Reference batch iterator has to be "
                            "an instance of BatchIterator!")
        self._ref_batch_iterator = batch_iterator
        self.batch_total = len(batch_iterator)
        self.global_total = self.cfg.epochs * len(batch_iterator)
        self.progress = tqdm(initial=self.global_step,
                             total=self.global_total)
        self.update_steps_according_to_ref_batch_iterator()

    def log_train_progress(self):
        self._print_line()
        log.info("| Name : %s | Epoch : %d/%d | Batches : %d/%d |"
                 % (self.cfg.name, self.epoch_step, self.epoch_total,
                    self.batch_step, self.batch_total))

    def _print_line(self, char='-', row=1, length=130):
        for i in range(row):
            log.info(char * length)

    def should_stop_training(self):
        self.update_steps_according_to_ref_batch_iterator()
        iterator = self.ref_batch_iterator
        return iterator.epoch_step >= self.epoch_total

    def update_steps_according_to_ref_batch_iterator(self):
        inc = self.ref_batch_iterator.global_step - self.global_step
        self.batch_step = self.ref_batch_iterator.batch_step
        self.epoch_step = self.ref_batch_iterator.epoch_step
        self.global_step = self.ref_batch_iterator.global_step
        self.progress.update(inc)
        self._update_gan_schedule()

    def increase_epoch_step(self):
        log.debug("Epoch %d stop! batch: %d"
                   % (self.epoch_step, self.batch_step))
        self.batch_step = 0
        self.epoch_step += 1
        self._update_gan_schedule  # at epoch [2, 4, 6]

    def save(self):
        self._save_step()
        self.net.save_modules()
        log.info("Model saved.")

    def load_if_exists(self):
        log_dir = listdir(self.cfg.log_dir)
        if (not any(fname.endswith('.ckpt') for fname in log_dir) or
            not (path.exists(self.step_fname))):
            log.info("Can't find {} and model.ckpt files. "
                     "Training begins from the scratch!"
                     .format(self.step_fname))
        else:
            log.info("Loading model from : %s" % (self.step_fname))
            self._load_step()
            self.net.load_modules()
            self._update_gan_schedule()

    def _save_step(self):
        with open(self.step_fname, 'w') as f:
            dump_dict = dict(global_step=self.global_step,
                             epoch_step=self.epoch_step,
                             batch_step=self.batch_step)
            json.dump(dump_dict, f, sort_keys=True, indent=4)

    def _load_step(self):
        with open(self.step_fname, 'r') as f:
            load_dict = json.load(f)
        log.info(load_dict)
        self.global_step = load_dict['global_step']
        self.epoch_step = load_dict['epoch_step']
        self.batch_step = load_dict['batch_step']

    def _init_gan_schedule(self):
        if self.cfg.niters_gan_schedule != "": # 2-4-6
            gan_schedule = self.cfg.niters_gan_schedule.split("-")
            gan_schedule =  [int(x) for x in gan_schedule]
            # for example: gan_schedule = [2, 4, 6]
        else:
            gan_schedule = []
        niters_gan = 1
        gan_schedule_list = []
        for i in range(1, self.cfg.epochs+1):
            if i in gan_schedule:
                niters_gan += 1
            gan_schedule_list.append(niters_gan)
        return gan_schedule_list

    def _update_gan_schedule(self):
        self.gan_iter = self.gan_schedule[self.epoch_step-1] # starts with 1
        # if self.gan_iter > self.gan_schedule[self.epoch_step-2]:
        #     log.info("GAN training loop schedule increased to : {}"
        #              "".format(self.niters_gan))
        # NOTE : make this to be called only when epoch is increased
