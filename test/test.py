import logging
import numpy as np
import os
import time
from datetime import datetime

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
from utils.utils import set_random_seed, to_gpu, set_logger

from test.evaluate_nltk import truncate, corp_bleu
from train.train_with_kenlm import train_lm, ids_to_sent_for_eval, \
                                print_ae_sents, print_gen_sents, \
                                load_test_data

"""
This function is for test session.
In test session, training is not performed.
To enter test session, append '--test' arg when running main file.
Test session assumes trained data, and user should provide the trained data directory via --name arg. ex:'--name train_data_name'
"""
def test(net):
    cfg = net.cfg # for brevity
    cfg.log_filepath = os.path.join(cfg.log_dir, "testlog.txt") # set output log file path
    #cfg, logger_name = 'main', filepath = cfg.log_filepath
    log = logging.getLogger('main')
    set_logger(cfg=cfg)
    log.info("test session {}".format(datetime.now()))
    sv = Supervisor(net)
    set_random_seed(cfg)
    fixed_noise = net.gen.make_noise(cfg, cfg.eval_size) # for generator
    test_sents = load_test_data(cfg)

    # exponentially decaying noise on autoencoder
    # noise_raius = 0.2(default)
    # noise_anneal = 0.995(default)
    net.ae.noise_radius = net.ae.noise_radius * cfg.noise_anneal

    #print status of data that is being used
    epoch = sv.epoch_step
    nbatch = sv.batch_step
    niter = sv.global_step
    print('epoch {}, nbatch {}, niter {}. \033[1;34m'.format(epoch, nbatch, niter)) # add color

    # test session
    while 1:
        # get the number of output sentence from AE
        print('default sample num:', cfg.log_nsample)
        ae_num = int(input("enter the number of AE sample(quit:0):")) # to quit test session, enter 0
        if ae_num == 0:
            break
        # Autoencoder
        batch = net.data_eval.next()
        tars, outs = eval_ae(cfg, net.ae, batch)
        # dump results
        print_ae_sents(net.vocab, tars, outs, batch.len, ae_num)

        # Generator + Discriminator_c
        fake_hidden = net.gen.generate(cfg, fixed_noise, False)
        ids_fake_eval, _ = net.ae.decode_only(cfg, fake_hidden,
                                              net.vocab, False)

        #get the number of output sentence from Generator
        gen_num = int(input("enter the number of generator sample(quit:0):"))
        if gen_num == 0:
            break
        # dump results
        print_gen_sents(net.vocab, ids_fake_eval, gen_num)

        # Discriminator_s
        if cfg.with_attn:
            a_real, a_fake = attns
            ids_tar = batch.tar.view(cfg.batch_size, -1).data.cpu().numpy()
            a_fake_r, a_fake_f = halve_attns(a_fake)
            print_attns(cfg, net.vocab,
                        dict(Real=(ids_tar, a_real),
                             Fake_R=(ids_real, a_fake_r),
                             Fake_F=(ids_fake, a_fake_f)))

        fake_sents = ids_to_sent_for_eval(net.vocab, ids_fake_eval)

        #choose range of evaluation
        eval_setting = input("Do you want to perform full evaluation?(y/n):")
        rp_scores = evaluate_sents(test_sents, fake_sents)

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
            save_path = "out/{}/niter{}_lm_generation".format(sv.cfg.name, niter), # .arpa file path
            n = cfg.N)
        log.info('Eval/bleu-4'+str(bleu4))
        log.info('Eval/6_Reverse_Perplexity'+str(ppl))
        ### end
    # end test session
    print('exit test' + '\033[0;0m') # reset color
