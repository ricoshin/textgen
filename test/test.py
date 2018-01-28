import collections
import logging
import numpy as np
import os
import time

from tensorboardX import SummaryWriter
import torch
import torch.nn as nn
from torch.autograd import Variable

from test.evaluate import evaluate_sents
from train.train_models import (train_ae, eval_ae_tf, eval_ae_fr, train_dec,
                                train_gen, train_disc_c, train_disc_s,
                                generate_codes, eval_gen_dec)
from train.train_helper import (load_test_data, append_pads, print_ae_tf_sents,
                                print_ae_fr_sents, print_gen_sents,
                                ids_to_sent_for_eval, halve_attns, print_attns)
from train.supervisor import Supervisor
from utils.utils import set_random_seed, to_gpu, set_logger
from datetime import datetime

from test.evaluate_nltk import truncate, corp_bleu
from train.train_with_kenlm import train_lm

from test.bleu_variation import leakgan_bleu, urop_bleu
from test.rouge import corp_rouge

dict = collections.OrderedDict
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
    testlog = logging.getLogger('test')
    set_logger(cfg=cfg, name='test')
    testlog.info("test session {}".format(datetime.now()))
    sv = Supervisor(net)
    set_random_seed(cfg)
    fixed_noise = net.gen.make_noise(cfg, cfg.eval_size) # for generator
    test_sents = load_test_data(cfg)

    # exponentially decaying noise on autoencoder
    # noise_raius = 0.2(default)
    # noise_anneal = 0.995(default) NOTE: fix this!
    net.enc.noise_radius = net.enc.noise_radius * cfg.noise_anneal

    #print status of data that is being used
    epoch = sv.epoch_step
    nbatch = sv.batch_step
    niter = sv.global_step
    print('epoch {}, nbatch {}, niter {}. \033[1;34m'.format(epoch, nbatch, niter)) # add color

    # get the number of batch size(num_samples)
    print('default batch size:', cfg.batch_size)
    batch_size = int(input("enter batch size(quit:0):")) # to quit test session, enter 0
    if batch_size ==0:
        return None
    net.cfg.batch_size = batch_size
    cfg.batch_size = batch_size

    # test session
    while 1:
        # get the number of output sentence from AE
        print('default sample num:', cfg.log_nsample)
        ae_num = int(input("enter the number of AE sample(quit:0):")) # to quit test session, enter 0
        if ae_num == 0:
            break

        # Autoencoder
        batch = net.data_eval.next()
        tars, outs = eval_ae_tf(net, batch)
        print_ae_tf_sents(net.vocab, tars, outs, batch.len, cfg.log_nsample)
        tars, outs = eval_ae_fr(net, batch)
        print_ae_fr_sents(net.vocab, tars, outs, cfg.log_nsample)

        # Generator + Discriminator_c
        ids_fake_eval = eval_gen_dec(cfg, net, fixed_noise)

        #get the number of output sentence from Generator
        gen_num = int(input("enter the number of generator sample(quit:0):"))
        if gen_num == 0:
            break
        # dump results
        print_gen_sents(net.vocab, ids_fake_eval, gen_num)

        # Discriminator_s
        """
        if cfg.with_attn:
            a_real, a_fake = attns
            ids_real, ids_fake = ids
            ids_fake_r = ids_fake[len(ids_fake)//2:]
            ids_fake_f = ids_fake[:len(ids_fake)//2]
            a_fake_r, a_fake_f = halve_attns(a_fake)
            print_attns(cfg, net.vocab,
                        dict(Real=(ids_real, a_real),
                             Fake_R=(ids_fake_r, a_fake_r),
                             Fake_F=(ids_fake_f, a_fake_f)))
        """
        fake_sents = ids_to_sent_for_eval(net.vocab, ids_fake_eval)

        #choose range of evaluation
        eval_setting = input("Do you want to perform full evaluation?(y/n):")
        rp_scores = evaluate_sents(test_sents, fake_sents)

        if eval_setting =='y' or eval_setting == 'Y': # full evaluation
            rouge = corp_rouge(references = test_sents, hypotheses=fake_sents)
            testlog.info('Eval/Rouge: '+str(rouge))
            bleu1 = corp_bleu(references=test_sents, hypotheses=fake_sents, gram=1)
            testlog.info('Eval/bleu-1: '+str(bleu1))
            bleu2 = corp_bleu(references=test_sents, hypotheses=fake_sents, gram=2)
            testlog.info('Eval/bleu-2: '+str(bleu2))
            bleu3 = corp_bleu(references=test_sents, hypotheses=fake_sents, gram=3)
            testlog.info('Eval/bleu-3: '+str(bleu3))
            bleu = corp_bleu(references=test_sents, hypotheses=fake_sents)
            #how to load pre-built arpa file?
            testlog.info('Eval/5_nltk_Bleu: '+str(bleu))
            testlog.info('Eval/leakgan_bleu: '+str(leakgan_bleu(test_sents, fake_sents)))
            testlog.info('Eval/urop_bleu: '+str(urop_bleu(test_sents, fake_sents)))

        bleu4 = corp_bleu(references=test_sents, hypotheses=fake_sents, gram=4)
        ppl = train_lm(eval_data=test_sents, gen_data = fake_sents,
            vocab = net.vocab,
            save_path = "out/{}/niter{}_lm_generation".format(sv.cfg.name, niter), # .arpa file path
            n = cfg.N)
        testlog.info('Eval/bleu-4: '+str(bleu4))
        testlog.info('Eval/6_Reverse_Perplexity: '+str(ppl))
    # end test session
    print('exit test' + '\033[0;0m') # reset color
