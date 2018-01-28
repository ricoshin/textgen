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
from utils.utils import set_random_seed, to_gpu

log = logging.getLogger('main')
dict = collections.OrderedDict

from test.evaluate_nltk import truncate, corp_bleu
from test.bleu_variation import leakgan_bleu, urop_bleu

"""
codes originally from ARAE : https://github.com/jakezhaojb/ARAE
some parts are modified
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
codes originally from ARAE
end here
"""


def train(net):
    log.info("Training start!")
    cfg = net.cfg # for brevity
    set_random_seed(cfg)
    fixed_noise = net.gen.make_noise(cfg, cfg.eval_size) # for generator
    writer = SummaryWriter(cfg.log_dir)
    sv = Supervisor(net)
    test_sents = load_test_data(cfg)

    while not sv.global_stop():
        while not sv.epoch_stop():

            # train autoencoder
            for i in range(cfg.niters_ae): # default: 1 (constant)
                if sv.epoch_stop():
                    break  # end of epoch
                batch = net.data_ae.next()
                rp_ae = train_ae(cfg, net.ae, batch)
                net.optim_enc.step()
                net.optim_dec.step()
                sv.inc_batch_step()

            # train gan
            for k in range(sv.gan_niter): # epc0=1, epc2=2, epc4=3, epc6=4

                # train discriminator/critic (at a ratio of 5:1)
                for i in range(cfg.niters_gan_d): # default: 5
                    # feed a seen sample within this epoch; good for early training
                    # randomly select single batch among entire batches in the epoch
                    batch = net.data_gan.next()

                    # train CodeDiscriminator
                    code_real, code_fake = generate_codes(cfg, net, batch)
                    rp_dc = train_disc_c(cfg, net, code_real, code_fake)
                    #err_dc_total, err_dc_real, err_dc_fake = err_dc

                    # train SampleDiscriminator
                    if cfg.with_attn and sv.epoch_step >= cfg.disc_s_hold:
                        ids_real, outs_real = \
                            net.ae.decode_only(cfg, code_real,net.vocab)
                        ids_fake, outs_fake = \
                            net.ae.decode_only(cfg, code_fake, net.vocab)
                        if cfg.disc_s_in == 'embed':
                            # "real" fake
                            outs_fake = torch.cat([outs_real, outs_fake], dim=0)
                            code_fake = torch.cat([code_real, code_fake], dim=0)
                            # "real" real
                            outs_real = batch.tar.view(cfg.batch_size, -1)
                            outs_real = append_pads(cfg, outs_real, net.vocab)
                            #outs_real = batch.tar.view(cfg.batch_size, -1)

                        rp_ds_l_gan, rp_ds_l_rec, rp_ds_pred, attns = \
                            train_disc_s(cfg, net.disc_s,outs_real, outs_fake,
                                         code_real, code_fake)

                        net.optim_disc_s.step()

                    net.optim_ae.step()
                    net.optim_disc_c.step()

                # train generator(with disc_c) / decoder(with disc_s)
                for i in range(cfg.niters_gan_g): # default: 1
                    rp_gen, code_fake = \
                        train_gen(cfg, net.gen, net.ae, net.disc_c)
                    if cfg.with_attn and sv.epoch_step >= cfg.disc_s_hold:
                        rp_dec = train_dec(cfg, net.ae, net.disc_s,
                                           code_fake, net.vocab)
                        net.optim_ae.step()

                    net.optim_gen.step()

            if not sv.batch_step % cfg.log_interval == 0:
                continue

            # exponentially decaying noise on autoencoder
            # noise_raius = 0.2(default)
            # noise_anneal = 0.995(default) NOTE: fix this!
            net.ae.noise_radius = net.ae.noise_radius * cfg.noise_anneal

            # Autoencoder
            batch = net.data_eval.next()
            tars, outs = eval_ae(cfg, net.ae, batch)
            # dump results
            rp_ae.drop_log_and_events(sv, writer)
            print_ae_sents(net.vocab, tars, outs, batch.len, cfg.log_nsample)

            # Generator + Discriminator_c
            fake_hidden = net.gen.generate(cfg, fixed_noise, False)
            ids_fake_eval, _ = net.ae.decode_only(cfg, fake_hidden,
                                                  net.vocab, False)
            # dump results
            rp_dc.update(dict(G=rp_gen.loss))
            rp_dc.drop_log_and_events(sv, writer, False)
            print_gen_sents(net.vocab, ids_fake_eval, cfg.log_nsample)

            # Discriminator_s
            if cfg.with_attn and sv.epoch_step >= cfg.disc_s_hold:
                rp_ds_l_gan.update(dict(G_Dec=rp_dec.loss))
                rp_ds_l_gan.drop_log_and_events(sv, writer)
                rp_ds_l_rec.drop_log_and_events(sv, writer, False)

                rp_ds_pred.update(dict(G_Dec=rp_dec.pred))
                rp_ds_pred.drop_log_and_events(sv, writer, False)

                a_real, a_fake = attns
                ids_tar = batch.tar.view(cfg.batch_size, -1).data.cpu().numpy()
                a_fake_r, a_fake_f = halve_attns(a_fake)
                print_attns(cfg, net.vocab,
                            dict(Real=(ids_tar, a_real),
                                 Fake_R=(ids_real, a_fake_r),
                                 Fake_F=(ids_fake, a_fake_f)))

            fake_sents = ids_to_sent_for_eval(net.vocab, ids_fake_eval)
            rp_scores = evaluate_sents(test_sents, fake_sents)
            rp_scores.drop_log_and_events(sv, writer, False)

            ### added by JWY
            if sv.batch_step % (2*cfg.log_interval) == 0:
                # lower reference size to lighten computation cost
                bleu = corp_bleu(references=test_sents[:min(len(fake_sents)*50, len(test_sents))],
                    hypotheses=fake_sents, gram=4)
                log.info('nltk bleu-{}: {}'.format(4, bleu))

                ppl = train_lm(eval_data=test_sents, gen_data = fake_sents,
                    vocab = net.vocab,
                    save_path = "out/{}/niter{}_lm_generation".format(sv.cfg.name, sv.batch_step),
                    n = cfg.N)
                log.info("Perplexity {}".format(ppl))
                rouge = copr_rouge(references = test_sents, hypotheses=fake_sents)
                log.info('Eval/Rouge: '+str(rouge))
                writer.add_scalar('Eval/4_rouge', rouge, sv.global_step)
                writer.add_scalar('Eval/5_nltk_Bleu', bleu, sv.global_step)
                writer.add_scalar('Eval/6_Reverse_Perplexity', ppl, sv.global_step)

                # leakgan, urop bleu
                leakgan = leakgan_bleu(test_sents[:min(len(fake_sents)*50, len(test_sents))], fake_sents)
                urop = urop_bleu(test_sents[:min(len(fake_sents)*50, len(test_sents))], fake_sents)
                log.info('leakgan_bleu: '+str(leakgan))
                log.info('urop_bleu: '+str(urop))
                writer.add_scalar('Eval/7_leakgan_bleu2', leakgan[0], sv.global_step)
                writer.add_scalar('Eval/7_leakgan_bleu3', leakgan[1], sv.global_step)
                writer.add_scalar('Eval/7_leakgan_bleu4', leakgan[2], sv.global_step)
                writer.add_scalar('Eval/8_urop_bleu2', urop[0], sv.global_step)
                writer.add_scalar('Eval/8_urop_bleu3', urop[1], sv.global_step)
                writer.add_scalar('Eval/8_urop_bleu4', urop[2], sv.global_step)
            ### end
            sv.save()

        # end of epoch ----------------------------
        sv.inc_epoch_step()
