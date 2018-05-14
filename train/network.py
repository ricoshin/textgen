from collections import OrderedDict
import logging
from os import path

import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from loader.data import (BatchCollator, POSBatchCollator, DataScheduler,
                         MyDataLoader)

from models.encoder import (EncoderRNN, EncoderCNN, CodeSmoothingRegularizer,
                            VariationalRegularizer)
from models.enc_disc import EncoderDiscModeWrapper, EncoderDisc
from models.decoder import DecoderRNN, DecoderCNN
from models.disc_code import CodeDiscriminator
from models.generator import Generator, ReversedGenerator
from models.disc_sample import SampleDiscriminator
from nn.embedding import Embedding

log = logging.getLogger('main')


class Network(object):
    """Instances of specific classes set as attributes in Network class
    will automatically be updated to the dictionaries as below:

    torch.nn.Module -> self._modules
    optim.Optimizer -> self._optimizers
    loader.corpus DataScheduler -> self._batch_schedulers

    """
    def __init__(self, cfg, corpus, vocab_word, vocab_tag=None):
        self.cfg = cfg
        self.corpus = corpus
        self.vocab_w = vocab_word
        self.vocab_t = vocab_tag
        self.ntokens = len(vocab_word)

        self._modules = OrderedDict()
        self._optimizers = OrderedDict()
        self._batch_schedulers = OrderedDict()

        self._build_dataset()
        self._build_network()
        self._build_optimizer()

    def __setattr__(self, name, value):
        object.__setattr__(self, name, value)

        if isinstance(value, nn.Module):
            self._check_init_by_name('_modules')
            self._modules[name] = value

        if isinstance(value, optim.Optimizer):
            self._check_init_by_name('_optimizers')
            name.replace('optim_', '')
            self._optimizers[name] = value

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
            collator = BatchCollator(cfg, vocab_word)

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

        if cfg.enc_type == 'cnn':
            Encoder = EncoderCNN
        elif cfg.enc_type == 'rnn':
            Encoder = EncoderRNN
        else:
            raise ValueError('Unknown encoder type!')

        if cfg.dec_type == 'cnn':
            Decoder = DecoderCNN
        elif cfg.dec_type == 'rnn':
            Decoder = DecoderRNN
        else:
            raise ValueError('Unknown decoder type!')

        # NOTE remove later!
        self.embed_w = Embedding(cfg, self.vocab_w)  # Word embedding
        self.enc = Encoder(cfg)  # Encoder
        self.reg = CodeSmoothingRegularizer(cfg)  # Code regularizer
        self.dec = Decoder(cfg, self.embed_w)  # Decoder
        self.dec2 = Decoder(cfg, self.embed_w)  # Decoder
        self.gen = Generator(cfg)  # Generator
        self.rev = ReversedGenerator(cfg)
        #self.disc = CodeDiscriminator(cfg, cfg.hidden_size_w)  # Discriminator
        #self.disc_s = SampleDiscriminator(cfg, cfg.hidden_size_w*2)

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
        self.optim_embed_w = optim_ae(self.embed_w)
        self.optim_enc = optim_ae(self.enc)
        self.optim_dec = optim_ae(self.dec)
        self.optim_dec2 = optim_ae(self.dec2)
        self.optim_reg = optim_ae(self.reg)
        #self.optim_reg_mu = optim_ae(self.reg.mu_layers)
        #self.optim_reg_sigma_ae = optim_ae(self.reg.sigma_layers)
        #self.optim_reg_sigma_gen = optim_gen(self.reg.sigma_layers)
        #self.optim_reg_gen = optim_gen(self.reg)
        self.optim_gen = optim_gen(self.gen)
        self.optim_rev = optim_gen(self.rev)
        #self.optim_disc = optim_disc(self.disc)

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

    def clip_grad_norm__by_names(self, *names):
        for name in names:
            module = self._modules.get(name, None)
            if module is None:
                raise ValueError("Can't find module name %s" % name)
            module.clip_grad_norm_()

    def step_optimizers_by_names(self, *names):
        for name in names:
            optim = self._optimizers.get(name, None)
            if optim is None:
                raise ValueError("Can't find optimizer name of %s" % name)
            optim.step()

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
