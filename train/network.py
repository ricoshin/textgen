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

        self.data_ae = BatchIterator(data_loader, cfg.cuda)
        self.data_gan = BatchIterator(data_loader, cfg.cuda)
        self.data_eval = BatchIterator(data_loader, cfg.cuda, volatile=True)
        #self.test_data_ae = BatchIterator(dataloder_ae_test)

        # Encoder/Discriminator (sample level)
        self.enc = EncoderDiscriminator(cfg, vocab)
        # Decoder/Generator (sample level)
        self.dec = DecoderRNN(cfg, vocab)
        # Generator (code level)
        self.gen = Generator(cfg)
        # Discriminator (code level)
        self.disc = CodeDiscriminator(cfg)

        # Print network modules
        log.info(self.enc)
        log.info(self.dec)
        log.info(self.gen)
        log.info(self.disc)

        # Optimizers
        params_enc = filter(lambda p: p.requires_grad, self.enc.parameters())
        params_dec = filter(lambda p: p.requires_grad, self.dec.parameters())

        self.optim_enc = optim.SGD(params_enc, lr=cfg.lr_ae) # default: 1
        self.optim_dec = optim.SGD(params_dec, lr=cfg.lr_ae) # default: 1
        self.optim_gen = optim.Adam(self.gen.parameters(),
                                    lr=cfg.lr_gan_g, # default: 0.00005
                                    betas=(cfg.beta1, 0.999))
        self.optim_disc = optim.Adam(self.disc.parameters(),
                                       lr=cfg.lr_gan_d, # default: 0.00001
                                       betas=(cfg.beta1, 0.999))

        if cfg.cuda:
            # self.ae = self.ae.cuda()
            self.enc = self.enc.cuda()
            self.dec = self.dec.cuda()
            self.gen = self.gen.cuda()
            self.disc = self.disc.cuda()
