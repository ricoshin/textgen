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
from train.train_models import (train_ae, eval_ae_tf, eval_ae_fr,
                                train_disc_ans, eval_disc_ans)
from train.train_helper import (load_test_data, append_pads, print_ae_tf_sents,
                                print_ae_fr_sents, print_gen_sents,
                                ids_to_sent_for_eval, halve_attns, print_attns)
from train.supervisor import Supervisor
from utils.utils import set_random_seed, to_gpu

log = logging.getLogger('main')
dict = collections.OrderedDict

from test.evaluate_nltk import truncate, corp_bleu
from test.bleu_variation import leakgan_bleu, urop_bleu

"""
codes originally from ARAE : https://github.com/jakezhaojb/ARAE
some parts are modified
"""
"""
from utils.utils_kenlm import train_ngram_lm, get_ppl
# save_path : save path of .arpa and .txt file
# N : N-gram language model. default 5.
def train_lm(eval_data, gen_data, vocab, save_path, n):
        #ppl = train_lm(eval_data=test_sents, gen_data = fake_sents,
        #    vocab = net.vocab,
        #    save_path = "out/{}/niter{}_lm_generation".format(sv.cfg.name, sv.batch_step),
        #    n = cfg.N)
    # input : test dataset
    #kenlm_path = '/home/jwy/venv/env36/lib/python3.5/site-packages/kenlm'
    kenlm_path = '/home/jwy/kenlm'
    #processing
    eval_sents = [truncate(s) for s in eval_data]
    gen_sents = [truncate(s) for s in gen_data]

    # write generated sentences to text file
    with open(save_path+".txt", "w") as f:
        # laplacian smoothing
        for word in vocab.word2idx.keys():
            if word == '<unk>' or word == '<eos>' or word == '<pad>':
                continue
            f.write(word+"\n")
        for sent in gen_sents:
            chars = " ".join(sent)
            f.write(chars+"\n")

    # train language model on generated examples
    lm = train_ngram_lm(kenlm_path=kenlm_path,
                        data_path=save_path+".txt",
                        output_path=save_path+".arpa",
                        N=n)
    # empty or too small .arpa file
    if lm == None:
        return 2147483647 # assign biggest value
    # evaluate
    ppl = get_ppl(lm, eval_sents)
    return ppl
"""
"""
codes originally from ARAE
end here
"""


def train(net):
    log.info("Training start!")
    cfg = net.cfg # for brevity
    set_random_seed(cfg)
    fixed_noise = net.gen.make_noise(cfg.eval_size) # for generator
    writer = SummaryWriter(cfg.log_dir)
    sv = Supervisor(net)
    test_q_sents, test_a_sents = load_test_data(cfg)

    while not sv.global_stop():
        while not sv.epoch_stop():

            # train autoencoder
            for i in range(cfg.niters_ae): # default: 1 (constant)
                if sv.epoch_stop():
                    break  # end of epoch
                ans_batch = net.data_ans.next() # answer batch
                batch = net.data_ae.next()
                rp_ae = train_ae(cfg, net, batch, ans_batch)
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
                    ans_batch = net.data_gan_ans.next()

                    # train
                    logit, loss = train_disc_ans(cfg, net, batch, ans_batch)

                    net.optim_disc_ans.step()

            if not sv.batch_step % cfg.log_interval == 0:
                continue

            # exponentially decaying noise on autoencoder
            # noise_raius = 0.2(default)
            # noise_anneal = 0.995(default) NOTE: fix this!
            net.enc.noise_radius = net.enc.noise_radius * cfg.noise_anneal

            # Autoencoder batch
            ans_batch = net.data_ans_eval.next()
            batch = net.data_eval.next()

            # make encoded answer embedding
            net.ans_enc.eval()
            ans_code = net.ans_enc(ans_batch.src, ans_batch.len, noise=True)

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
            log.info('targets: {}'.format(targets))
            log.info('outputs: {}'.format(outputs))
            writer.add_scalar('Disc_Ans/1_loss', loss, sv.global_step)
            sv.save()

        # end of epoch ----------------------------
        sv.inc_epoch_step()
