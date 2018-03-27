from collections import OrderedDict
import logging
import numpy as np
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from models.decoder import WordIdTranscriber
from test.evaluate import evaluate_sents
from train.train_helper import load_test_data, mask_output_target
from train.supervisor import TrainingSupervisor
from utils.writer import ResultWriter
from utils.utils import set_random_seed, to_gpu

log = logging.getLogger('main')
odict = OrderedDict


class Trainer(object):
    def __init__(self, net):
        log.info("Training start!")
        #set_random_seed(net.cfg)

        self.net = net
        self.cfg = net.cfg
        #self.fixed_noise = net.gen.make_noise_size_of(net.cfg.eval_size)

        self.test_sents = load_test_data(net.cfg)
        self.pos_one = to_gpu(net.cfg.cuda, torch.FloatTensor([1]))
        self.neg_one = self.pos_one * (-1)

        self.result = ResultWriter(net.cfg)
        self.sv = TrainingSupervisor(net, self.result)
        #self.sv.interval_func_train.update({net.enc.decay_noise_radius: 200})

        while not self.sv.is_end_of_training():
            self.train_loop(self.cfg, self.net, self.sv)

    def train_loop(self, cfg, net, sv):
        """Main training loop"""

        with sv.training_context():

            # train autoencoder
            for i in range(sv.niter_ae):  # default: 1 (constant)
                if net.data_ae.step.is_end_of_step(): break
                batch = net.data_ae.next()
                self._train_autoencoder(batch)

            # train gan
            # for k in range(sv.niter_gan): # epc0=1, epc2=2, epc4=3, epc6=4
            #
            #     # train discriminator/critic (at a ratio of 5:1)
            #     for i in range(cfg.niter_gan_d): # default: 5
            #         batch = net.data_gan.next()
            #         self._train_discriminator(batch)
            #         #self._train_encoder(batch)
            #
            #     # train generator(with disc_c) / decoder(with disc_s)
            #     for i in range(cfg.niter_gan_g): # default: 1
            #         self._train_generator()
            #         #self._train_regularizer(batch)

        if sv.is_evaluation():
            with sv.evaluation_context():
                batch = net.data_eval.next()
                #self._eval_autoencoder(batch, 'tf')
                self._eval_autoencoder(batch)
                # self._generate_text()

    def _train_autoencoder(self, batch, name='AE_train'):
        self.net.set_modules_train_mode(True)

        # Build graph
        embed = self.net.embed(batch.src)
        code = self.net.enc.with_noise(embed, batch.len)

        code_var = self.net.reg.with_var(code)
        cos_sim = F.cosine_similarity(code, code_var, dim=1).mean()
        decoded = self.net.dec(code_var)

        # Register hook
        code_var.register_hook(self.net.enc.save_ae_grad_norm_hook)
        decoded.embed.register_hook(self.net.dec.save_ae_grad_norm_hook)

        # Compute word prediction loss and accuracy
        prob_flat = decoded.prob.view(-1, self.cfg.vocab_size_w)
        #import pdb; pdb.set_trace()
        loss = self.net.dec.criterion_nll(prob_flat, batch.tar)

        _, max_ids = torch.max(prob_flat, 1)
        acc = torch.mean(max_ids.eq(batch.tar).float())
        loss.backward()

        # optimize
        self.net.optim_embed.step()
        self.net.optim_enc.step()
        self.net.optim_reg_ae.step()
        self.net.optim_dec.step()

        self.result.add(name, odict(
            loss=loss.data[0],
            acc=acc.data[0],
            cosim=cos_sim.data[0],
            var=self.net.reg.var,
            noise=self.net.enc.noise_radius,
            text=decoded.get_text_with_target(batch.src),
            ))

    def _eval_autoencoder(self, batch, name='AE_eval'):
        self.net.set_modules_train_mode(True)

        # Build graph
        embed = self.net.embed(batch.src)
        code = self.net.enc(embed, batch.len)
        code_var = self.net.reg.with_var(code)
        cos_sim = F.cosine_similarity(code, code_var, dim=1).mean()
        decoded = self.net.dec(code_var)

        code_embed = ResultWriter.Embedding(
            embed=code.data, text=decoded.get_text_batch())
        code_var_embed = ResultWriter.Embedding(
            embed=code_var.data, text=decoded.get_text_batch())

        # Compute word prediction loss and accuracy
        masked_output, masked_target = \
            mask_output_target(decoded.prob, batch.tar, self.cfg.vocab_size_w)
        loss = self.net.dec.criterion_nll(masked_output, masked_target)
        _, max_ids = torch.max(masked_output, 1)
        acc = torch.mean(max_ids.eq(masked_target).float())

        self.result.add(name, odict(
            code=code_embed,
            code_var=code_var_embed,
            loss=loss.data[0],
            acc=acc.data[0],
            cosim=cos_sim.data[0],
            var=self.net.reg.var,
            noise=self.net.enc.noise_radius,
            text=decoded.get_text_with_target(batch.src),
            ))

    def _train_regularizer(self, batch, name="Logvar_train"):
        self.net.set_modules_train_mode(True)

        # Build graph
        embed = self.net.embed(batch.src)
        code_real = self.net.enc(embed, batch.len)
        code_real_var = self.net.reg.with_var(code_real)
        disc_real = self.net.disc_c(code_real_var)

        # loss / backprop
        disc_real.backward(self.neg_one)
        self.net.optim_reg_gen.step()

        self.result.add(name, odict(loss=disc_real.data[0]))

    def _train_encoder(self, batch, name="Enc_train"):
        self.net.set_modules_train_mode(True)

        # Code generation
        embed = self.net.embed(batch.src)
        code_real = self.net.enc(embed, batch.len)
        disc_real = self.net.disc_c(code_real)

        disc_real.backward(self.neg_one)
        self.net.optim_gen_c.step()

        self.result.add(name, odict(loss=disc_real.data[0]))

    def _train_generator(self, name="Gen_train"):
        self.net.set_modules_train_mode(True)

        # Build graph
        code_fake = self.net.gen.for_train()
        disc_fake = self.net.disc_c(code_fake) # NOTE batch norm should be on

        # loss / backprop
        disc_fake.backward(self.pos_one)
        self.net.optim_gen_c.step()

        self.result.add(name, odict(loss=disc_fake.data[0]))

    def _train_discriminator(self, batch, name="Disc_train"):
        self.net.set_modules_train_mode(True)

        # Code generation
        embed = self.net.embed(batch.src)
        code_real = self.net.enc(embed, batch.len)
        code_real_var = self.net.reg.with_var(code_real)
        code_fake = self.net.gen.for_train()

        # Grad hook : gradient scaling
        code_real_var.register_hook(self.net.enc.scale_disc_grad_hook)

        # Weight clamping for WGAN
        self.net.disc_c.clamp_weights()

        disc_real = self.net.disc_c(code_real_var)
        disc_fake = self.net.disc_c(code_fake.detach())
        loss_total = disc_real - disc_fake

        # WGAN backward
        disc_real.backward(self.pos_one)
        disc_fake.backward(self.neg_one)
        #loss_total.backward()

        #self.net.optim_embed.step() #NOTE
        self.net.optim_enc.step()
        self.net.optim_reg_ae.step()
        self.net.optim_disc_c.step()

        self.result.add(name, odict(
            loss_toal=loss_total.data[0],
            loss_real=disc_real.data[0],
            loss_fake=disc_fake.data[0],
            ))

    def _generate_text(self, name="Generated"):
        self.net.set_modules_train_mode(True)

        # Build graph
        code_fake = self.net.gen.for_eval()
        decoded = self.net.dec(code_fake)

        code_fake_embed = ResultWriter.Embedding(
            embed=code_fake.data, text=decoded.get_text_batch())

        self.result.add(name, odict(
            code=code_fake_embed,
            text=decoded.get_text(),
            ))

        # Evaluation
        scores = evaluate_sents(self.test_sents, decoded.get_text())
        self.result.add("Evaluation", scores)
