import collections
import logging
import numpy as np
import os
import time

from tensorboardX import SummaryWriter
import torch
import torch.nn as nn
from torch.autograd import Variable

from train.train_models import (train_ae, eval_ae_tf, eval_ae_fr,
                                train_disc_ans, eval_disc_ans)
from train.train_helper import (load_test_data, append_pads, print_ae_tf_sents,
                                print_ae_fr_sents, print_gen_sents,
                                ids_to_sent_for_eval, halve_attns, print_attns)
from train.supervisor import Supervisor
from utils.utils import set_random_seed, to_gpu

log = logging.getLogger('main')
dict = collections.OrderedDict


def train(net):
    log.info("Training start!")
    cfg = net.cfg # for brevity
    set_random_seed(cfg)
    writer = SummaryWriter(cfg.log_dir)
    sv = Supervisor(net)
    test_q_sents, test_a_sents = load_test_data(cfg)

    while not sv.global_stop():
        while not sv.epoch_stop():

            # train autoencoder
            for i in range(cfg.niters_ae): # default: 1 (constant)
                if sv.epoch_stop():
                    break  # end of epoch
                batch = net.data_ae.next()
                rp_ae = train_ae(cfg, net, batch)
                net.optim_ans_enc.step()
                net.optim_enc.step()
                net.optim_dec.step()
                sv.inc_batch_step()

            # train discriminator
            for k in range(sv.gan_niter): # epc0=1, epc2=2, epc4=3, epc6=4
                # train discriminator/critic (at a ratio of 5:1)
                for i in range(cfg.niters_gan_d): # default: 5
                    # feed a seen sample within this epoch; good for early training
                    # randomly select single batch among entire batches in the epoch
                    batch = net.data_gan.next()

                    # train
                    logit, loss = train_disc_ans(cfg, net, batch)

                    net.optim_disc_ans.step()

            if not sv.batch_step % cfg.log_interval == 0:
                continue

            # exponentially decaying noise on autoencoder
            # noise_raius = 0.2(default)
            # noise_anneal = 0.995(default) NOTE: fix this!
            net.enc.noise_radius = net.enc.noise_radius * cfg.noise_anneal

            # Autoencoder batch
            batch = net.data_eval.next()

            # make encoded answer embedding
            net.ans_enc.eval()
            ans_code = net.ans_enc(batch.a, batch.a_len, noise=True)

            # Autoencoder eval
            tars, outs = eval_ae_tf(net, batch, ans_code)
            print_ae_tf_sents(net.q_vocab, tars, outs, batch.len, cfg.log_nsample)
            tars, outs = eval_ae_fr(net, batch, ans_code)
            print_ae_fr_sents(net.q_vocab, tars, outs, cfg.log_nsample)

            # dump results
            rp_ae.drop_log_and_events(sv, writer)
            #print_ae_sents(net.vocab, tar)

            # Answer Discriminator
            logit, loss, targets, outputs = eval_disc_ans(net, batch, ans_code)

            # dump results
            log.info('loss: {}'.format(loss))
            #log.info('targets: {}'.format(targets))
            #log.info('outputs: {}'.format(outputs))
            writer.add_scalar('Disc_Ans/1_loss', loss, sv.global_step)
            sv.save()

        # end of epoch ----------------------------
        sv.inc_epoch_step()
