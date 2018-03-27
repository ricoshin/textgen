from collections import OrderedDict
import logging
from os import path

import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from loader.data import (BatchCollator, POSBatchCollator, DataScheduler,
                         MyDataLoader)

from models.encoder import EncoderRNN, EncoderCNN, CodeSmoothingRegularizer
from models.enc_disc import EncoderDiscModeWrapper, EncoderDisc
from models.decoder import DecoderRNN, DecoderCNN
from models.disc_code import CodeDiscriminator
from models.generator import Generator
from models.disc_sample import SampleDiscriminator
from nn.embedding import Embedding

log = logging.getLogger('main')


class Network(object):
    """Instances of specific classes set as attributes in this classes
    will automatically be updated to predefined dictionaries as below:

    loader.corpus DataScheduler -> self._batch_schedulers
    torch.nn.Module -> self._modules

    """
    def __init__(self, cfg, corpus, vocab_word, vocab_tag=None):
        self.cfg = cfg
        self.corpus = corpus
        self.vocab_w = vocab_word
        self.vocab_t = vocab_tag
        self.ntokens = len(vocab_word)

        self._modules = OrderedDict()
        self._batch_schedulers = OrderedDict()

        self._build_dataset()
        self._build_network()
        self._build_optimizer()

    def __setattr__(self, name, value):
        object.__setattr__(self, name, value)

        if isinstance(value, nn.Module):
            self._check_init_by_name('_modules')
            self._modules[name] = value

        if isinstance(value, DataScheduler):
            self._check_init_by_name('_batch_schedulers')
            self._batch_schedulers[name] = value

    def _build_dataset(self):
        cfg = self.cfg
        corpus = self.corpus
        vocab_word = self.vocab_w
        vocab_tag = self.vocab_t

        if cfg.pos_tag:
            collator = POSBatchCollator(cfg, vocab_word, vocab_tag)
        else:
            collator = BatchCollator(cfg, vocab)

        self.data_train = MyDataLoader(corpus, cfg.batch_size, shuffle=True,
                                       num_workers=0, collate_fn=collator,
                                       drop_last=True, pin_memory=True)
        self.data_eval = MyDataLoader(corpus, cfg.eval_size, shuffle=True,
                                      num_workers=0, collate_fn=collator,
                                      drop_last=True, pin_memory=True)

        self.data_ae = DataScheduler(cfg, self.data_train)
        self.data_gan = DataScheduler(cfg, self.data_train)
        self.data_eval = DataScheduler(cfg, self.data_eval, volatile=True)
        #self.test_data_ae = BatchIterator(dataloder_ae_test)

    def _build_network(self):
        cfg = self.cfg

        # NOTE remove later!
        self.vocab_t._embed_init = None
        self.vocab_t.embed_size = cfg.embed_size_t
        self.vocab_t._generate_embedding()

        self.embed_t = Embedding(cfg, self.vocab_t)
        self.embed_w = Embedding(cfg, self.vocab_w)  # Word embedding
        self.enc = EncoderCNN(cfg)  # Encoder
        self.reg = CodeSmoothingRegularizer(cfg)  # Code regularizer
        self.dec = DecoderRNN(cfg, self.embed_w, self.embed_t)  # Decoder
        self.gen = Generator(cfg)  # Generator
        self.disc_t = CodeDiscriminator(cfg, self.cfg.hidden_size_t)  # Syntactic D
        self.disc_w = CodeDiscriminator(cfg, self.cfg.hidden_size_w)  # Semantic D

        self._print_modules_info()
        if cfg.cuda:
            self._upload_modules_to_gpu()

    def _build_optimizer(self):
        optim_ae = lambda module : optim.SGD(module.parameters(),
                                             lr=self.cfg.lr_ae)
        optim_gen = lambda module : optim.Adam(module.parameters(),
                                               lr=self.cfg.lr_gan_g,
                                               betas=(self.cfg.beta1, 0.999))
        optim_disc = lambda module : optim.Adam(module.parameters(),
                                                lr=self.cfg.lr_gan_d,
                                                betas=(self.cfg.beta1, 0.999))
        # Optimizers
        self.optim_embed_t = optim_ae(self.embed_t)
        self.optim_embed_w = optim_ae(self.embed_w)
        self.optim_enc = optim_ae(self.enc)
        self.optim_dec = optim_ae(self.dec)
        self.optim_reg_ae = optim_ae(self.reg)
        self.optim_reg_gen = optim_gen(self.reg)
        self.optim_gen = optim_gen(self.gen)
        self.optim_disc_t = optim_disc(self.disc_t)
        self.optim_disc_w = optim_disc(self.disc_w)

    def _print_modules_info(self):
        for name, module in self.registered_modules():
            log.info(module)

    def _upload_modules_to_gpu(self):
        for name, module in self.registered_modules():
            module = module.cuda()

    def registered_modules(self):
        self._check_init_by_name('_modules')
        for name, module in self._modules.items():
            yield name, module

    def registered_batch_schedulers(self):
        self._check_init_by_name('_batch_schedulers')
        for name, module in self._batch_schedulers.items():
            yield name, module

    def save_modules(self):
        self._check_init_by_name('_modules')
        for name, module in self.registered_modules():
            fname = path.join(self.cfg.log_dir, name + '.ckpt')
            with open(fname, 'wb') as f:
                torch.save(module.state_dict(), f)

    def load_modules(self):
        self._check_init_by_name('_modules')
        for name, module in self.registered_modules():
            fname = path.join(self.cfg.log_dir, name + '.ckpt')
            module.load_state_dict(torch.load(fname))
            log.info('Module has been loaded from : %s' % fname)

    def save_batch_schedulers(self):
        self._check_init_by_name('_batch_schedulers')
        for name, scheduler in self.registered_batch_schedulers():
            fname = path.join(self.cfg.log_dir, name + '.pickle')
            scheduler.save_as_pickle(fname)

    def load_batch_schedulers(self):
        self._check_init_by_name('_batch_schedulers')
        for name, scheduler in self.registered_batch_schedulers():
            fname = path.join(self.cfg.log_dir, name + '.pickle')
            scheduler.load_from_pickle(fname)
            log.info('BatchIterator has been loaded from : %s' % fname)

    def set_modules_train_mode(self, train_mode):
        self._check_init_by_name('_modules')
        for name, module in self.registered_modules():
            if train_mode:
                module = module.trainer
            else:
                module = module.tester

    def _check_init_by_name(self, name):
        if not name in self.__dict__:
            raise AttributeError(
                "Cannot assign modules before Network.__init__() call")
