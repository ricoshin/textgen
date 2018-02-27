from collections import OrderedDict
import logging
from os import path

import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader

from loader.corpus import BatchingDataset, BatchingPOSDataset, BatchIterator
from models.encoder import EncoderRNN, EncoderCNN, CodeSmoothingRegularizer
from models.enc_disc import EncoderDiscModeWrapper, EncoderDisc
from models.decoder import DecoderRNN, DecoderCNN
from models.disc_code import CodeDiscriminator
from models.generator import Generator
from models.disc_sample import SampleDiscriminator
from nn.embedding import WordEmbedding

log = logging.getLogger('main')


class Network(object):
    def __init__(self, cfg, corpus, vocab, vocab_pos=None):
        self.cfg = cfg
        self.corpus = corpus
        self.vocab = vocab
        self.vocab_pos = vocab_pos
        self.ntokens = len(vocab)

        self._modules = OrderedDict()
        self._build_dataset()
        self._build_network()
        self._build_optimizer()

    def __setattr__(self, name, value):
        object.__setattr__(self, name, value)
        if isinstance(value, nn.Module):
            self._check_module_dict_init()
            self._modules[name] = value

    def _build_dataset(self):
        cfg = self.cfg
        corpus = self.corpus
        vocab = self.vocab
        vocab_pos = self.vocab_pos

        if cfg.pos_tag:
            batching_dataset = BatchingPOSDataset(cfg, vocab, vocab_pos)
        else:
            batching_dataset = BatchingDataset(cfg, vocab)

        data_loader = DataLoader(corpus, cfg.batch_size, shuffle=True,
                                 num_workers=0, collate_fn=batching_dataset,
                                 drop_last=True, pin_memory=True)

        self.data_ae = BatchIterator(data_loader, cfg.cuda)
        self.data_gan = BatchIterator(data_loader, cfg.cuda)
        self.data_eval = BatchIterator(data_loader, cfg.cuda, volatile=True)
        #self.test_data_ae = BatchIterator(dataloder_ae_test)

    def _build_network(self):
        cfg = self.cfg

        self.embed = WordEmbedding(cfg, self.vocab) # Word embedding
        self.enc = EncoderCNN(cfg) # Encoder
        self.reg = CodeSmoothingRegularizer(cfg) # Code regularizer
        self.dec = DecoderRNN(cfg, self.embed) # Decoder
        self.gen = Generator(cfg) # Generator
        self.disc_c = CodeDiscriminator(cfg) # Discriminator - code level

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
        self.optim_embed = optim_ae(self.embed)
        self.optim_enc = optim_ae(self.enc)
        self.optim_dec = optim_ae(self.dec)
        self.optim_reg = optim_ae(self.reg)
        self.optim_mu = optim_ae(self.reg.fc_mu)
        self.optim_logvar = optim_gen(self.reg.fc_logvar)
        self.optim_gen_s = optim_gen(self.dec)
        self.optim_gen_c = optim_gen(self.gen)

        self.optim_disc_c = optim_disc(self.disc_c)

    def _print_modules_info(self):
        for name, module in self.registered_modules():
            log.info(module)

    def _upload_modules_to_gpu(self):
        for name, module in self.registered_modules():
            module = module.cuda()

    def registered_modules(self):
        self._check_module_dict_init()
        for name, module in self._modules.items():
            yield name, module

    def save_modules(self):
        self._check_module_dict_init()
        for name, module in self.registered_modules():
            fname = path.join(self.cfg.log_dir, name + '.ckpt')
            with open(fname, 'wb') as f:
                torch.save(module.state_dict(), f)

    def load_modules(self):
        self._check_module_dict_init()
        for name, module in self.registered_modules():
            fname = path.join(self.cfg.log_dir, name + '.ckpt')
            module.load_state_dict(torch.load(fname))

    def _check_module_dict_init(self):
        if not '_modules' in self.__dict__:
            raise AttributeError(
                "Cannot assign modules before Network.__init__() call")
