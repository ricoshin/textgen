import collections
import logging
import numpy as np
import os
import time

from tensorboardX import SummaryWriter
import torch
import torch.nn as nn
from torch.autograd import Variable

from train.train_models import (train_ae, eval_ae_tf, eval_ae_fr)
from train.train_helper import (load_test_data, append_pads, print_ae_tf_sents,
                                print_ae_fr_sents, print_gen_sents,
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

    # end test session
    print('exit test' + '\033[0;0m') # reset color
