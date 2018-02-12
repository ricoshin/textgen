import logging

import torch.optim as optim
from torch.utils.data import DataLoader

from loader.corpus import BatchingDataset, BatchingPOSDataset, BatchIterator
from models.convnets import CNNArchitect, EncoderCNN, DecoderCNN
from models.encoder import EncoderRNN
from models.enc_disc import EncoderDisc, EncoderDiscModeWrapper
from models.decoder import DecoderRNN
from models.disc_code import CodeDiscriminator
from models.generator import Generator
from models.disc_sample import SampleDiscriminator
from nn.embedding import WordEmbedding

log = logging.getLogger('main')


class Network(object):
    def __init__(self, cfg, corpus, vocab, vocab_pos=None):
        self.cfg = cfg
        self.vocab = vocab
        self.ntokens = len(vocab)

        if cfg.pos_tag:
            batching_dataset = BatchingPOSDataset(cfg, vocab, vocab_pos)
        else:
            batching_dataset = BatchingDataset(cfg, vocab)

        self.cfg.max_len += 1 # NOTE

        data_loader = DataLoader(corpus, cfg.batch_size, shuffle=True,
                                 num_workers=0, collate_fn=batching_dataset,
                                 drop_last=True, pin_memory=True)

        self.data_ae = BatchIterator(data_loader, cfg.cuda)
        self.data_gan = BatchIterator(data_loader, cfg.cuda)
        self.data_eval = BatchIterator(data_loader, cfg.cuda, volatile=True)
        #self.test_data_ae = BatchIterator(dataloder_ae_test)

        # Word embedding
        self.embed = WordEmbedding(cfg, vocab)
        # CNN architecture
        arch_cnn = CNNArchitect(cfg)
        # Encoder
        self.enc = EncoderCNN(cfg, arch_cnn)
        # Decoder
        self.dec = DecoderCNN(cfg, arch_cnn)
        # Generator
        self.gen = Generator(cfg)
        # Discriminator - code level
        self.disc_c = CodeDiscriminator(cfg)
        # Discriminator - sample level
        self.disc_s = CodeDiscriminator(cfg)
        #self.disc_g = CodeDiscriminator(cfg)

        # Print network modules
        log.info(self.embed)
        log.info(self.enc)
        log.info(self.dec)
        log.info(self.gen)
        log.info(self.disc_c)
        log.info(self.disc_s)
        #log.info(self.disc_g)

        # Optimizers
        params_enc = filter(lambda p: p.requires_grad, self.enc.parameters())
        params_dec = filter(lambda p: p.requires_grad, self.dec.parameters())
        #params_disc_d = filter(lambda p: p.requires_grad, self.disc_d.parameters())
        #params_disc_g = filter(lambda p: p.requires_grad, self.disc_g.parameters())

        self.optim_embed = optim.SGD(self.embed.parameters(), lr=cfg.lr_ae) # default: 1
        self.optim_enc = optim.SGD(params_enc, lr=cfg.lr_ae) # default: 1
        self.optim_dec = optim.SGD(params_dec, lr=cfg.lr_ae) # default: 1
        self.optim_gen = optim.Adam(self.gen.parameters(),
                                    lr=cfg.lr_gan_g, # default: 0.00005
                                    betas=(cfg.beta1, 0.999))
        self.optim_disc_c = optim.Adam(self.disc_c.parameters(),
                                       lr=cfg.lr_gan_d, # default: 0.00001
                                       betas=(cfg.beta1, 0.999))
        self.optim_disc_s = optim.Adam(self.disc_s.parameters(),
                                       lr=cfg.lr_gan_d, # default: 0.00001
                                       betas=(cfg.beta1, 0.999))
        # self.optim_disc_g = optim.Adam(params_disc_g,
        #                                lr=cfg.lr_gan_d, # default: 0.00001
        #                                betas=(cfg.beta1, 0.999))

        if cfg.cuda:
            self.embed = self.embed.cuda()
            self.enc = self.enc.cuda()
            self.dec = self.dec.cuda()
            self.gen = self.gen.cuda()
            self.disc_c = self.disc_c.cuda()
            self.disc_s = self.disc_s.cuda()
            #self.disc_g = self.disc_g.cuda()
