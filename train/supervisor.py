import json
import logging
from os import path
from tqdm import tqdm

import torch

log = logging.getLogger('main')

class Supervisor(object):
    def __init__(self, net):
        self.cfg = net.cfg
        self.net = net

        # all steps should start from 0
        self.global_step = 0
        self.epoch_step = 0
        self.batch_step = 0

        self.global_total = self.cfg.epochs * len(net.data_ae)
        self.epoch_total = self.cfg.epochs
        self.batch_total = len(net.data_ae)
        self.gan_schedule = self._init_gan_schedule()
        self.gan_niter = self.gan_schedule[0]

        self.step_fname = path.join(self.cfg.log_dir, 'schedule.json')
        self.enc_fname = path.join(self.cfg.log_dir, 'enc.ckpt')
        self.dec_fname = path.join(self.cfg.log_dir, 'dec.ckpt')
        self.gen_fname = path.join(self.cfg.log_dir, 'gen.ckpt')
        self.disc_c_fname = path.join(self.cfg.log_dir, 'disc_c.ckpt')
        if self.cfg.with_attn and not self.cfg.enc_disc:
            self.disc_s_fname = path.join(self.cfg.log_dir, 'disc_s.ckpt')

        self.load()
        self.progress = tqdm(initial=self.global_step, total=self.global_total)

    def inc_batch_step(self):
        self.batch_step += 1
        self.global_step += 1
        self.progress.update(1)

    def inc_epoch_step(self):
        log.debug("Epoch %d stop! batch: %d"
                   % (self.epoch_step, self.batch_step))
        self.batch_step = 1
        self.epoch_step += 1
        self.net.data_ae.reset()
        self._update_gan_schedule  # at epoch [2, 4, 6]

    def save(self):
        self._save_step()
        self._save_model()
        log.info("Model saved.")

    def load(self):
        if not (path.exists(self.step_fname) and path.exists(self.enc_fname)):
            log.info("Can't find {} and model.ckpt files"
                     .format(self.step_fname))
        else:
            log.info("Loading model from : %s" % (self.step_fname))
            self._load_step()
            self._load_model()
            self._update_gan_schedule()

    def global_stop(self):
        return self.epoch_step > self.epoch_total

    def epoch_stop(self):
        return self.batch_step > self.batch_total

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

    def _save_model(self):
        log_dir = self.cfg.log_dir
        with open(self.enc_fname, 'wb') as f:
            torch.save(self.net.enc.state_dict(), f)
        with open(self.dec_fname, 'wb') as f:
            torch.save(self.net.dec.state_dict(), f)
        with open(self.gen_fname, 'wb') as f:
            torch.save(self.net.gen.state_dict(), f)
        with open(self.disc_c_fname, 'wb') as f:
            torch.save(self.net.disc_c.state_dict(), f)
        if self.cfg.with_attn and not self.cfg.enc_disc:
            with open(self.disc_s_fname, 'wb') as f:
                torch.save(self.net.disc_s.state_dict(), f)

    def _load_model(self):
        self.net.enc.load_state_dict(torch.load(self.enc_fname))
        self.net.dec.load_state_dict(torch.load(self.dec_fname))
        self.net.gen.load_state_dict(torch.load(self.gen_fname))
        self.net.disc_c.load_state_dict(torch.load(self.disc_c_fname))
        if self.cfg.with_attn and not self.cfg.enc_disc:
            self.net.disc_s.load_state_dict(torch.load(self.disc_s_fname))

    def _init_gan_schedule(self):
        if self.cfg.niters_gan_schedule != "": # 2-4-6
            gan_schedule = self.cfg.niters_gan_schedule.split("-")
            gan_schedule =  [int(x) for x in gan_schedule]
            # for example: gan_schedule = [2, 4, 6]
        else:
            gan_schedule = []
        gan_niter = 1
        gan_schedule_list = []
        for i in range(1, self.cfg.epochs+1):
            if i in gan_schedule:
                gan_niter += 1
            gan_schedule_list.append(gan_niter)
        return gan_schedule_list

    def _update_gan_schedule(self):
        self.gan_iter = self.gan_schedule[self.epoch_step-1] # starts with 1
        if self.gan_iter > self.gan_schedule[self.epoch_step-2]:
            log.info("GAN training loop schedule increased to : {}"
                     "".format(self.gan_niter))
        else:
            log.info("GAN training loop schedule remains constant : {}"
                     "".format(self.gan_iter))
