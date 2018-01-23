import logging

import torch.optim as optim
from torch.utils.data import DataLoader

from loader.book_corpus import BatchingDataset, BatchIterator
#from models.autoencoder import Autoencoder
from models.encoder import EncoderRNN
from models.enc_disc import EncoderDiscriminator
from models.decoder import DecoderRNN
from models.disc_code import CodeDiscriminator
from models.generator import Generator
from models.disc_sample import SampleDiscriminator

log = logging.getLogger('main')


class Network(object):
    def __init__(self, cfg, book_corpus, vocab):
        self.cfg = cfg
        self.vocab = vocab
        self.ntokens = len(vocab)

        batching_dataset = BatchingDataset(vocab)
        data_loader = DataLoader(book_corpus, cfg.batch_size, shuffle=True,
                                 num_workers=0, collate_fn=batching_dataset,
                                 drop_last=True, pin_memory=True)

        #dataloader_ae_test = DataLoader(book_corpus, cfg.batch_size,
        #                                shuffle=False, num_workers=4,
        #                                collate_fn=batching_dataset)
        self.data_ae = BatchIterator(data_loader, cfg.cuda)
        self.data_gan = BatchIterator(data_loader, cfg.cuda)
        self.data_eval = BatchIterator(data_loader, cfg.cuda, volatile=True)
        #self.test_data_ae = BatchIterator(dataloder_ae_test)

        # Autoencoder
        # self.ae = Autoencoder(cfg, vocab.embed_mat)

        # Encoder
        if cfg.enc_disc_share:
            self.enc = EncoderRNN(cfg, vocab)
        else:
            self.enc = SampleDiscriminator(cfg, vocab)
        # Decoder
        self.dec = DecoderRNN(cfg, vocab)
        # Generator
        self.gen = Generator(cfg)
        # Discriminator - code level
        self.disc_c = CodeDiscriminator(cfg)
        # Discriminator - sample level
        if cfg.with_attn:
            self.disc_s = SampleDiscriminator(cfg, vocab)

        # Print network modules
        log.info(self.enc)
        log.info(self.dec)
        log.info(self.gen)
        log.info(self.disc_c)
        if cfg.with_attn:
            log.info(self.disc_s)

        # Optimizers
        # params_ae = filter(lambda p: p.requires_grad, self.ae.parameters())
        params_enc = filter(lambda p: p.requires_grad, self.enc.parameters())
        params_dec = filter(lambda p: p.requires_grad, self.dec.parameters())

        #self.optim_ae = optim.SGD(params_ae, lr=cfg.lr_ae) # default: 1
        self.optim_enc = optim.SGD(params_enc, lr=cfg.lr_ae) # default: 1
        self.optim_dec = optim.SGD(params_dec, lr=cfg.lr_ae) # default: 1
        self.optim_gen = optim.Adam(self.gen.parameters(),
                                    lr=cfg.lr_gan_g, # default: 0.00005
                                    betas=(cfg.beta1, 0.999))
        self.optim_disc_c = optim.Adam(self.disc_c.parameters(),
                                       lr=cfg.lr_gan_d, # default: 0.00001
                                       betas=(cfg.beta1, 0.999))
        if cfg.with_attn:
            params_disc_s = filter(lambda p: p.requires_grad,
                                   self.disc_s.parameters())
            self.optim_disc_s = optim.Adam(params_disc_s,
                                           lr=cfg.lr_gan_d, # default: 0.00001
                                           betas=(cfg.beta1, 0.999))

        if cfg.cuda:
            # self.ae = self.ae.cuda()
            self.enc = self.enc.cuda()
            self.dec = self.dec.cuda()
            self.gen = self.gen.cuda()
            self.disc_c = self.disc_c.cuda()
            if cfg.with_attn:
                self.disc_s = self.disc_s.cuda()
