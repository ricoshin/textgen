import json
import logging
from os import path
from tqdm import tqdm

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from autoencoder import Autoencoder
from book_corpus import BatchingDataset
from code_disc import CodeDiscriminator
from generator import Generator
from sample_disc import SampleDiscriminator

log = logging.getLogger('main')


class BatchIterator(object):
    def __init__(self, dataloader):
        self.__dataloader = dataloader
        self.__batch_iter = iter(self.__dataloader)
        self.__batch = None # initial value

    def __len__(self):
        return len(self.__dataloader)

    @property
    def batch(self):
        return self.__batch

    def reset(self):
        self.__batch_iter = iter(self.__dataloader)

    def next_or_none(self):
        self.__batch = next(self.__batch_iter, None)
        return self.__batch

    def next(self):
        self.next_or_none()
        if self.__batch is None:
            self.reset()
            self.__batch = next(self.__batch_iter)
        return self.__batch


class Network(object):
    def __init__(self, cfg, book_corpus, vocab):
        self.cfg = cfg
        self.vocab = vocab
        self.ntokens = len(vocab)

        batching_dataset = BatchingDataset(vocab)
        dataloader_ae = DataLoader(book_corpus, cfg.batch_size, shuffle=True,
                                   num_workers=0, collate_fn=batching_dataset,
                                   drop_last=True, pin_memory=True)
        dataloader_gan = DataLoader(book_corpus, cfg.batch_size, shuffle=True,
                                    num_workers=0, collate_fn=batching_dataset,
                                    drop_last=True, pin_memory=True)

        #dataloader_ae_test = DataLoader(book_corpus, cfg.batch_size,
        #                                shuffle=False, num_workers=4,
        #                                collate_fn=batching_dataset)
        self.data_ae = BatchIterator(dataloader_ae)
        self.data_gan = BatchIterator(dataloader_gan)
        #self.test_data_ae = BatchIterator(dataloder_ae_test)

        # Autoencoder
        self.ae = Autoencoder(cfg, vocab.embed_mat)
        # Generator
        self.gen = Generator(ninput=cfg.z_size,
                             noutput=cfg.hidden_size,
                             layers=cfg.arch_g,
                             gpu=cfg.cuda)
        # Discriminator - code level
        self.disc_c = CodeDiscriminator(ninput=cfg.hidden_size,
                                        noutput=1,
                                        layers=cfg.arch_d,
                                        gpu=cfg.cuda)
        # Discriminator - sample level
        if cfg.with_attn:
            self.disc_s = SampleDiscriminator(cfg, vocab.embed_mat)

        # Print network modules
        log.info(self.ae)
        log.info(self.gen)
        log.info(self.disc_c)
        if cfg.with_attn:
            log.info(self.disc_s)
            params_disc_s = filter(lambda p: p.requires_grad,
                                   self.disc_s.parameters())
        # Optimizers
        params_ae = filter(lambda p: p.requires_grad, self.ae.parameters())
        #params_gen = filter(lambda p: p.requires_grad, self.gen.parameters())
        #params_disc_c = filter(lambda p: p.requires_grad,
        #                       self.disc_c.parameters())

        self.optim_ae = optim.SGD(params_ae, lr=cfg.lr_ae) # default: 1
        self.optim_gen = optim.Adam(self.gen.parameters(),
                                    lr=cfg.lr_gan_g, # default: 0.00005
                                    betas=(cfg.beta1, 0.999))
        self.optim_disc_c = optim.Adam(self.disc_c.parameters(),
                                       lr=cfg.lr_gan_d, # default: 0.00001
                                       betas=(cfg.beta1, 0.999))
        if cfg.with_attn:
            self.optim_disc_s = optim.Adam(params_disc_s,
                                           lr=cfg.lr_gan_d, # default: 0.00001
                                           betas=(cfg.beta1, 0.999))

        if cfg.cuda:
            self.ae = self.ae.cuda()
            self.gen = self.gen.cuda()
            self.disc_c = self.disc_c.cuda()
            if cfg.with_attn:
                self.disc_s = self.disc_s.cuda()


class TrainingSupervisor(object):
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
        self.ae_fname = path.join(self.cfg.log_dir, 'ae.ckpt')
        self.gen_fname = path.join(self.cfg.log_dir, 'gen.ckpt')
        self.disc_c_fname = path.join(self.cfg.log_dir, 'disc_c.ckpt')
        if self.cfg.with_attn:
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
        if not (path.exists(self.step_fname) and path.exists(self.ae_fname)):
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
        with open(self.ae_fname, 'wb') as f:
            torch.save(self.net.ae.state_dict(), f)
        with open(self.gen_fname, 'wb') as f:
            torch.save(self.net.gen.state_dict(), f)
        with open(self.disc_c_fname, 'wb') as f:
            torch.save(self.net.disc_c.state_dict(), f)
        if self.cfg.with_attn:
            with open(self.disc_s_fname, 'wb') as f:
                torch.save(self.net.disc_s.state_dict(), f)

    def _load_model(self):
        self.net.ae.load_state_dict(torch.load(self.ae_fname))
        self.net.gen.load_state_dict(torch.load(self.gen_fname))
        self.net.disc_c.load_state_dict(torch.load(self.disc_c_fname))
        if self.cfg.with_attn:
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
