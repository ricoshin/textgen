import logging

import torch.optim as optim
from torch.utils.data import DataLoader

from loader.simple_questions import BatchingDataset, BatchIterator
from models.encoder import Encoder
from models.decoder import Decoder
from models.disc_answer import AnswerDiscriminator
log = logging.getLogger('main')


class Network(object):
    def __init__(self, cfg, train_data, q_vocab, a_vocab):
        self.cfg = cfg
        self.q_vocab = q_vocab
        self.a_vocab = a_vocab
        self.q_ntokens = len(q_vocab)
        self.a_ntokens = len(a_vocab)

        batching_dataset = BatchingDataset(q_vocab)
        print("train data len: ", len(train_data))
        #train_data_loader = DataLoader(train_data, cfg.batch_size, shuffle=True,
        #                         num_workers=0, collate_fn=batching_dataset,
        #                         drop_last=True, pin_memory=True)
        train_data_loader = DataLoader(train_data, cfg.batch_size, shuffle=True,
                                 num_workers=0, collate_fn=q_batching_dataset,
                                 drop_last=True, pin_memory=True)
        eval_data_loader = DataLoader(train_data, cfg.eval_size, shuffle=True,
                                 num_workers=0, collate_fn=a_batching_dataset,
                                 drop_last=True, pin_memory=True)
        #dataloader_ae_test = DataLoader(book_corpus, cfg.batch_size,
        #                                shuffle=False, num_workers=4,
        #                                collate_fn=batching_dataset)
        self.data_ae = BatchIterator(train_data_loader, cfg.cuda)
        self.data_gan = BatchIterator(train_data_loader, cfg.cuda)
        self.data_eval = BatchIterator(eval_data_loader, cfg.cuda, volatile=True)
        #self.test_data_ae = BatchIterator(dataloder_ae_test)

        # Encoder for answer
        self.ans_enc = Encoder(cfg, a_vocab)
        # Autoencoder
        self.enc = Encoder(cfg, q_vocab)
        self.dec = Decoder(cfg, q_vocab)
        # Answer Discriminator
        self.disc_ans = AnswerDiscriminator(cfg, q_vocab)

        # Print network modules
        log.info(self.enc)
        log.info(self.dec)
        log.info(self.ans_enc)
        log.info(self.disc_ans)

        # Optimizers
        params_ans_enc = filter(lambda p: p.requires_grad, self.ans_enc.parameters())
        params_enc = filter(lambda p: p.requires_grad, self.enc.parameters())
        params_dec = filter(lambda p: p.requires_grad, self.dec.parameters())
        params_disc_ans = filter(lambda p: p.requires_grad, self.disc_ans.parameters())

        self.optim_ans_enc = optim.SGD(params_ans_enc, lr=cfg.lr_ae)
        self.optim_enc = optim.SGD(params_enc, lr=cfg.lr_ae) # default: 1
        self.optim_dec = optim.SGD(params_dec, lr=cfg.lr_ae) # default: 1
        self.optim_disc_ans = optim.Adam(params_disc_ans,
                                        lr=cfg.lr_gan_d,
                                        betas=(cfg.beta1, 0.999))

        if cfg.cuda:
            self.ans_enc = self.ans_enc.cuda()
            self.enc = self.enc.cuda()
            self.dec = self.dec.cuda()
            self.disc_ans = self.disc_ans.cuda()

class Network_Separated(object):
    def __init__(self, cfg, train_q_data, train_a_data, q_vocab, a_vocab):
        self.cfg = cfg
        self.q_vocab = q_vocab
        self.a_vocab = a_vocab
        self.q_ntokens = len(q_vocab)
        self.a_ntokens = len(a_vocab)

        q_batching_dataset = BatchingDataset(q_vocab)
        a_batching_dataset = BatchingDataset(a_vocab)
        print("question train data len: ", len(train_q_data))
        print("answer train data len: ", len(train_a_data))
        #train_data_loader = DataLoader(train_data, cfg.batch_size, shuffle=True,
        #                         num_workers=0, collate_fn=batching_dataset,
        #                         drop_last=True, pin_memory=True)
        train_q_data_loader = DataLoader(train_q_data, cfg.batch_size, shuffle=True,
                                 num_workers=0, collate_fn=q_batching_dataset,
                                 drop_last=True, pin_memory=True)
        train_a_data_loader = DataLoader(train_a_data, cfg.batch_size, shuffle=True,
                                 num_workers=0, collate_fn=a_batching_dataset,
                                 drop_last=True, pin_memory=True)
        eval_a_data_loader = DataLoader(train_a_data, cfg.eval_size, shuffle=True,
                                 num_workers=0, collate_fn=a_batching_dataset,
                                 drop_last=True, pin_memory=True)
        #dataloader_ae_test = DataLoader(book_corpus, cfg.batch_size,
        #                                shuffle=False, num_workers=4,
        #                                collate_fn=batching_dataset)
        self.data_ae_ans = BatchIterator(train_a_data_loader, cfg.cuda)
        self.data_ae = BatchIterator(train_q_data_loader, cfg.cuda)
        self.data_gan = BatchIterator(train_q_data_loader, cfg.cuda)
        self.data_gan_ans = BatchIterator(train_a_data_loader, cfg.cuda)
        self.data_eval = BatchIterator(train_q_data_loader, cfg.cuda, volatile=True)
        self.data_eval_ans = BatchIterator(eval_a_data_loader, cfg.cuda, volatile=True)
        #self.test_data_ae = BatchIterator(dataloder_ae_test)

        # Encoder for answer
        self.ans_enc = Encoder(cfg, a_vocab)
        # Autoencoder
        self.enc = Encoder(cfg, q_vocab)
        self.dec = Decoder(cfg, q_vocab)
        # Answer Discriminator
        self.disc_ans = AnswerDiscriminator(cfg, q_vocab)

        # Print network modules
        log.info(self.enc)
        log.info(self.dec)
        log.info(self.ans_enc)
        log.info(self.disc_ans)

        # Optimizers
        params_ans_enc = filter(lambda p: p.requires_grad, self.ans_enc.parameters())
        params_enc = filter(lambda p: p.requires_grad, self.enc.parameters())
        params_dec = filter(lambda p: p.requires_grad, self.dec.parameters())
        params_disc_ans = filter(lambda p: p.requires_grad, self.disc_ans.parameters())

        self.optim_ans_enc = optim.SGD(params_ans_enc, lr=cfg.lr_ae)
        self.optim_enc = optim.SGD(params_enc, lr=cfg.lr_ae) # default: 1
        self.optim_dec = optim.SGD(params_dec, lr=cfg.lr_ae) # default: 1
        self.optim_disc_ans = optim.Adam(params_disc_ans,
                                        lr=cfg.lr_gan_d,
                                        betas=(cfg.beta1, 0.999))

        if cfg.cuda:
            self.ans_enc = self.ans_enc.cuda()
            self.enc = self.enc.cuda()
            self.dec = self.dec.cuda()
            self.disc_ans = self.disc_ans.cuda()
