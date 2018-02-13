import collections
import logging
import numpy as np
import os
import time

from tensorboardX import SummaryWriter
import torch
import torch.nn as nn
from torch.autograd import Variable

from train.train_models import (train_ae, eval_ae_tf, eval_ae_fr, eval_gen_dec,
                                eval_disc_ans)
from train.train_helper import (load_test_data, append_pads, print_ae_tf_sents,
                                print_ae_fr_sents, print_gen_sents, print_gen_sents_test,
                                ids_to_sent_for_eval, halve_attns, print_attns)
from train.supervisor import Supervisor
from utils.utils import set_random_seed, to_gpu, set_logger
from datetime import datetime

dict = collections.OrderedDict
"""
This function is for test session.
In test session, training is not performed.
To enter test session, append '--test' arg when running main file.
Test session assumes trained data, and user should provide the trained data directory via --name arg. ex:'--name train_data_name'
"""
def test(net):
    cfg = net.cfg # for brevity
    # setting logger
    cfg.log_filepath = os.path.join(cfg.log_dir, "testlog.txt") # set output log file path
    #cfg, logger_name = 'main', filepath = cfg.log_filepath
    testlog = logging.getLogger('test')
    set_logger(cfg=cfg, name='test')
    testlog.info("test session {}".format(datetime.now()))

    sv = Supervisor(net)
    set_random_seed(cfg)
    fixed_noise = net.gen.make_noise(cfg.eval_size) # for generator
    test_q_sents, test_a_sents = load_test_data(cfg)

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


    # get question batch
    batch = net.data_eval.next()
    # test session
    while 1:
        # get the number of output sentence from AE
        print('default sample num:', cfg.log_nsample)
        ae_num = int(input("enter the number of AE sample(quit:0):")) # to quit test session, enter 0
        if ae_num == 0:
            break

        # get answer batch
        a_batch = net.data_eval.next()
        
        # encoded answer
        net.ans_enc.eval()
        ans_code = net.ans_enc(a_batch.a, a_batch.a_len, noise=True, ispacked=False)

        # Autoencoder eval
        ae_eval = input('want to evaluate autoencoder?(y/n)')
        if ae_eval == 'y' or ae_eval == 'Y':
            tars, outs = eval_ae_tf(net, batch, ans_code)
            print_ae_tf_sents(net.q_vocab, tars, outs, batch.q_len, cfg.log_nsample)
            tars, outs = eval_ae_fr(net, batch, ans_code)
            print_ae_fr_sents(net.q_vocab, tars, outs, cfg.log_nsample)

        # generator + discriminator_c
        ids_fake_eval = eval_gen_dec(cfg, net, fixed_noise, ans_code) # return generated fake ids
        
        # dump results
        print_gen_sents_test(net.q_vocab, net.a_vocab, ids_fake_eval, a_batch, cfg.log_nsample)

        fake_sents = ids_to_sent_for_eval(net.q_vocab, ids_fake_eval)

        # Answer Discriminator
        disc_ans_eval = input('want to evaluate disc_ans?(y/n)')
        if disc_ans_eval == 'y' or disc_ans_eval == 'Y':
            logit, loss, targets, outputs = eval_disc_ans(net, batch, ans_code)

            # dump results
            testlog.info('disc_ans loss: {}'.format(loss.data))
            testlog.info('targets: {}'.format(ids_to_sent_for_eval(net.a_vocab, targets)))
            testlog.info('outputs: {}'.format(ids_to_sent_for_eval(net.a_vocab, outputs)))

    # end test session
    print('exit test' + '\033[0;0m') # reset color
