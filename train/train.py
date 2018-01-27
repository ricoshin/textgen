import logging
import numpy as np
import os
import time

from tensorboardX import SummaryWriter
import torch
import torch.nn as nn
from torch.autograd import Variable

from models.autoencoder import Autoencoder
from models.code_disc import CodeDiscriminator
from models.generator import Generator
from models.sample_disc import SampleDiscriminator

from test.evaluate import evaluate_sents
from train.train_models import (train_ae, eval_ae_tf, eval_ae_fr, train_dec,
                                train_gen, train_disc_c, train_disc_s)
from train.train_helper import (load_test_data, append_pads, print_ae_tf_sents,
                                print_ae_fr_sents, print_gen_sents,
                                ids_to_sent_for_eval, halve_attns, print_attns)
from train.supervisor import Supervisor
from utils.utils import set_random_seed, to_gpu

log = logging.getLogger('main')


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
                net.optim_ae.step()
                sv.inc_batch_step()

            # train gan
            for k in range(sv.gan_niter): # epc0=1, epc2=2, epc4=3, epc6=4

                # train discriminator/critic (at a ratio of 5:1)
                for i in range(cfg.niters_gan_d): # default: 5
                    # feed a seen sample within this epoch; good for early training
                    # randomly select single batch among entire batches in the epoch
                    batch = net.data_gan.next()

                    # train CodeDiscriminator
                    code_real = net.ae.encode_only(cfg, batch)
                    code_fake = net.gen.generate(cfg, None, False)
                    rp_dc = train_disc_c(cfg, net.disc_c, net.ae,
                                         code_real, code_fake)
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
            tars, outs = eval_ae_tf(cfg, net.ae, batch)
            print_ae_tf_sents(net.vocab, tars, outs, batch.len, cfg.log_nsample)
            tars, outs = eval_ae_fr(cfg, net.ae, batch, net.vocab)
            print_ae_fr_sents(net.vocab, tars, outs, cfg.log_nsample)

            # dump results
            rp_ae.drop_log_and_events(sv, writer)
            #print_ae_sents(net.vocab, tar)

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

            sv.save()

        # end of epoch ----------------------------
        sv.inc_epoch_step()
