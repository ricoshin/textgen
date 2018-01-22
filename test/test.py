import logging
import numpy as np
import os
import time

import torch
import torch.nn as nn


from models.autoencoder import Autoencoder
from models.code_disc import CodeDiscriminator
from models.generator import Generator
from models.sample_disc import SampleDiscriminator

from test.evaluate import evaluate_sents
from train.train_models import (train_ae, eval_ae, train_dec, train_gen,
                                        train_disc_c, train_disc_s)
from train.train_helper import (load_test_data, append_pads, print_ae_sents,
                                        print_gen_sents, ids_to_sent_for_eval,
                                        halve_attns, print_attns)
from train.supervisor import Supervisor
from utils.utils import set_random_seed, to_gpu

from train_helper import TrainingSupervisor

from test.evaluate_nltk import truncate, corp_bleu
from train.train_with_kenlm import train_lm, ids_to_sent_for_eval, \
                                print_ae_sents, print_gen_sents, \
                                load_test_data


log = logging.getLogger('main')

def test(net):
    log = logging.getLogger('main')
    cfg.log_filepath = os.path.join(cfg.log_dir, "testlog.txt")
    #cfg, logger_name = 'main', filepath = cfg.log_filepath
    set_logger(cfg=cfg)
    log.info("test session {}".format(datetime.now()))
    cfg = net.cfg # for brevity
    sv = TrainingSupervisor(net)
    set_random_seed(cfg)
    fixed_noise = Generator.make_noise_(cfg, cfg.eval_size) # for generator
    test_sents = load_test_data(cfg)

    # exponentially decaying noise on autoencoder
    # noise_raius = 0.2(default)
    # noise_anneal = 0.995(default)
    net.ae.noise_radius = net.ae.noise_radius * cfg.noise_anneal

    epoch = sv.epoch_step
    nbatch = sv.batch_step
    niter = sv.global_step
    print('epoch {}, nbatch {}, niter {}. \033[94m'.format(epoch, nbatch, niter))

    net.data_ae.reset()
    batch = net.data_ae.next()

    #test session
    while 1:
        # Autoencoder
        print('default sample num:'+'\033[94m', cfg.log_nsample)
        ae_num = int(input("enter the number of AE sample(quit:0):"))
        if ae_num == 0:
            break
        tars, outs = Autoencoder.eval_(cfg, net.ae, batch)
        #print_nums(sv, 'AutoEnc', dict(Loss=ae_loss,
        #                               Accuracy=ae_acc))
        print_ae_sents(net.vocab, tars, outs, batch.len, ae_num)

        real_code = Autoencoder.encode_(cfg, net.ae, batch)
        real_ids, real_outs = \
            Autoencoder.decode_(cfg, net.ae, real_code, net.vocab)
        # Generator + Discriminator_c
        fake_hidden = Generator.generate_(cfg, net.gen, fixed_noise, False)
        fake_ids, _ = Autoencoder.decode_(cfg, net.ae, fake_hidden,
                                          net.vocab, False)
        #print_nums(sv, 'CodeGAN', dict(Loss_D_Total=err_d_c,
        #                               Loss_D_Real=err_d_c_real,
        #                               Loss_D_Fake=err_d_c_fake,
        #                               Loss_G=err_g))
        gen_num = int(input("enter the number of generator sample(quit:0):"))
        if gen_num == 0:
            break

        print_gen_sents(net.vocab, fake_ids, gen_num)

        # Discriminator_s
        # attns is output of discriminator train function
        #if cfg.with_attn and epoch >= cfg.disc_s_hold:
        #    print_attns(cfg, net.vocab, real_ids, fake_ids, *attns)

        fake_sents = ids_to_sent_for_eval(net.vocab, fake_ids)

        ### added by JWY
        eval_setting = input("Do you want to perform full evaluation?(y/n):")
        scores = evaluate_sents(test_sents, fake_sents)
        log.info(scores) # NOTE: change later!
        if eval_setting =='y' or eval_setting == 'Y': # full evaluation
            bleu1 = corp_bleu(references=test_sents, hypotheses=fake_sents, gram=1)
            bleu2 = corp_bleu(references=test_sents, hypotheses=fake_sents, gram=2)
            bleu3 = corp_bleu(references=test_sents, hypotheses=fake_sents, gram=3)
            bleu = corp_bleu(references=test_sents, hypotheses=fake_sents)
            #how to load pre-built arpa file?
            log.info('Eval/bleu-1'+str(bleu1))
            log.info('Eval/bleu-2'+str(bleu2))
            log.info('Eval/bleu-3'+str(bleu3))
            log.info('Eval/5_nltk_Bleu'+str(bleu))

        bleu4 = corp_bleu(references=test_sents, hypotheses=fake_sents, gram=4)
        ppl = train_lm(eval_data=test_sents, gen_data = fake_sents,
            vocab = net.vocab,
            save_path = "out/{}/niter{}_lm_generation".format(cfg.name, niter),
            n = cfg.N)
        log.info('Eval/bleu-4'+str(bleu4))
        log.info('Eval/6_Reverse_Perplexity'+str(ppl))
        ### end
    print('exit test' + '\033[97m')
