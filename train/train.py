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
from train.train_models import (train_ae, eval_ae_tf, eval_ae_fr, train_gen_s,
                                train_gen, train_disc_c, train_disc_s,
                                generate_codes, eval_gen_dec, train_enc,
                                train_exposure, recon_code_fake)
from train.train_helper import (ResultPackage, load_test_data,
                                print_ae_tf_sents, print_ae_fr_sents,
                                print_gen_sents, ids_to_sent_for_eval,
                                halve_attns, print_attns)
from train.supervisor import TrainingSupervisor
from utils.utils import set_random_seed, to_gpu

log = logging.getLogger('main')
dict = collections.OrderedDict

def train(net):
    log.info("Training start!")
    cfg = net.cfg # for brevity
    set_random_seed(cfg)
    fixed_noise = net.gen.make_noise(cfg.eval_size) # for generator
    writer = SummaryWriter(cfg.log_dir)
    sv = TrainingSupervisor(net)
    sv.register_ref_batch_iterator(net.data_ae)
    test_sents = load_test_data(cfg)

    while not sv.should_stop_training():

        # train autoencoder
        for i in range(sv.niters_ae): # default: 1 (constant)
            if sv.should_stop_training():
                break  # end of epoch
            batch = net.data_ae.next()

            # Embedding
            embed = net.embed.trainer(batch.src)
            # Encoder
            code = net.enc.trainer(embed)
            # Regularizer
            code_var = net.reg.trainer.with_var(code)
            # Decoder
            decoded = net.dec.trainer.teacher_forcing(code_var, batch)

            # Gradient saving
            code_var.register_hook(net.enc.save_ae_grad_norm_hook)
            decoded.embed.register_hook(net.dec.save_ae_grad_norm_hook)

            # Train
            rp_ae = train_ae(cfg, net, batch, decoded)

        # train gan
        for k in range(sv.niters_gan): # epc0=1, epc2=2, epc4=3, epc6=4

            # train discriminator/critic (at a ratio of 5:1)
            for i in range(cfg.niters_gan_d): # default: 5
                batch = net.data_gan.next()

                # Code generation
                embed = net.embed.trainer(batch.src)
                code_real = net.enc.trainer(embed)
                code_real_x = net.reg.trainer.without_var(code_real)
                code_fake = net.gen.tester()

                # weight clamping for WGAN
                net.disc_c = net.disc_c.clamp_weights()

                # code real
                disc_real = net.disc_c.trainer(code_real_x)
                disc_fake = net.disc_c.trainer(code_fake.detach())

                # Gradient scaling
                code_real.register_hook(net.enc.scale_disc_grad_hook)

                # Train
                rp_dc = train_disc_c(cfg, net, disc_real, disc_fake)

            # train generator(with disc_c) / decoder(with disc_s)
            for i in range(cfg.niters_gan_g): # default: 1
                # embed = net.embed.trainer(batch.src)
                # code_real = net.enc.trainer(embed)
                # code_real_o = net.reg.trainer.with_var(code_real)
                # disc_real = net.disc_c.trainer(code_real_o)
                # rp_enc = train_enc(cfg, net, disc_real)

                # train code generator
                code_fake = net.gen.trainer()
                disc_fake = net.disc_c.trainer(code_fake) # NOTE batch norm should be on
                rp_gen = train_gen(cfg, net, disc_fake)


        if not sv.global_step % cfg.log_interval == 0:
            continue

        # exponentially decaying noise on autoencoder
        # noise_raius = 0.2(default)
        # noise_anneal = 0.995(default) NOTE: fix this!
        # if sv.global_step % 200 == 0:
        #     net.enc.noise_radius = net.enc.noise_radius * cfg.noise_anneal


        # Autoencoder
        sv.log_train_progress()
        batch = net.data_eval.next()

        rp_ae.update(dict(Noise_radius=net.enc.noise_radius))
        rp_ae.drop_log_and_event(sv, writer)
        tars, outs = eval_ae_tf(net, batch)
        print_ae_tf_sents(net.vocab, tars, outs, batch.len, cfg.log_nsample)

        tars, outs = eval_ae_fr(net, batch)
        print_ae_fr_sents(net.vocab, tars, outs, cfg.log_nsample)

        rp_dc.update(dict(G_Loss=rp_gen.loss)) # NOTE : mismatch
        rp_dc.drop_log_and_event(sv, writer, False)
        ids_fake_eval = eval_gen_dec(cfg, net, fixed_noise)
        print_gen_sents(net.vocab, ids_fake_eval, 9)

        # dump results

        #print_ae_sents(net.vocab, tar)

        # rp_ae_fr.drop_log_and_event(sv, writer)
        # Generator + Discriminator_c

        # dump results


        # # Discriminator_s
        # if cfg.with_attn and sv.epoch_step >= cfg.disc_s_hold:
        #     rp_ds_loss.update(dict(G_Dec=#rp_dec.loss))
        #     rp_ds_loss.drop_log_and_event#s(sv, writer)
        #     rp_ds_pred.update(dict(G_Dec=#rp_dec.pred))
        #     rp_ds_pred.drop_log_and_event#s(sv, writer, False)
        #
        #     a_real, a_fake = attns
        #     ids_real, ids_fake = ids
        #     ids_fake_r = ids_fake[len(ids_fake)//2:]
        #     ids_fake_f = ids_fake[:len(ids_fake)//2]
        #     a_fake_r, a_fake_f = halve_attns(a_fake)
        #     print_attns(cfg, net.vocab,
        #                 dict(Real=(ids_real, a_real),
        #                      Fake_R=(ids_fake_r, a_fake_r),
        #                      Fake_F=(ids_fake_f, a_fake_f)))

        fake_sents = ids_to_sent_for_eval(net.vocab, ids_fake_eval)
        rp_scores = evaluate_sents(test_sents, fake_sents)
        rp_scores.drop_log_and_event(sv, writer, False)

        sv.save()
