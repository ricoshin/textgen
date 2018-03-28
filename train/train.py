import logging
import os
import time
from collections import OrderedDict
from test.evaluate import evaluate_sents

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from train.supervisor import TrainingSupervisor
from train.train_helper import (GradientScalingHooker, load_test_data,
                                mask_output_target)
from utils.utils import set_random_seed, to_gpu
from utils.writer import ResultWriter

log = logging.getLogger('main')
odict = OrderedDict


class Trainer(object):
    def __init__(self, net):
        log.info("Training start!")
        # set_random_seed(net.cfg)
        self.net = net
        self.cfg = net.cfg
        #self.fixed_noise = net.gen.make_noise_size_of(net.cfg.eval_size)

        self.test_sents = load_test_data(net.cfg)
        self.pos_one = to_gpu(net.cfg.cuda, torch.FloatTensor([1]))
        self.neg_one = self.pos_one * (-1)

        self.result = ResultWriter(net.cfg)
        self.sv = TrainingSupervisor(net, self.result)
        #self.sv.interval_func_train.update({net.enc.decay_noise_radius: 200})

        self.hooker = GradientScalingHooker(net.cfg.gan_to_enc)

        while not self.sv.is_end_of_training():
            self.train_loop(self.cfg, self.net, self.sv)

    def train_loop(self, cfg, net, sv):
        """Main training loop"""

        with sv.training_context():

            # train autoencoder
            for i in range(sv.niter_ae):  # default: 1 (constant)
                if net.data_ae.step.is_end_of_step():
                    break
                batch = net.data_ae.next()
                self._train_autoencoder(batch)

            # train gan
            for k in range(sv.niter_gan):  # epc0=1, epc2=2, epc4=3, epc6=4

                # train discriminator/critic (at a ratio of 5:1)
                for i in range(cfg.niter_gan_d):  # default: 5
                    batch = net.data_gan.next()
                    self._train_discriminator(batch)
                    # self._train_regularizer(batch)

                # train generator(with disc_c) / decoder(with disc_s)
                for i in range(cfg.niter_gan_g):  # default: 1
                    self._train_generator()

        if sv.is_evaluation():
            with sv.evaluation_context():
                batch = net.data_eval.next()
                #self._eval_autoencoder(batch, 'tf')
                self._eval_autoencoder(batch, 'tf')
                self._generate_text()

    def _train_autoencoder(self, batch, name='AE_train'):
        self.net.set_modules_train_mode(True)

        # Build graph
        embed = self.net.embed_w(batch.src)
        code = self.net.enc(embed, batch.len)
        #code_var = self.net.reg.with_var(code)
        #cos_sim = F.cosine_similarity(code, code_var, dim=1).mean()
        decoded = self.net.dec.teacher_forcing(code, batch)
        # tags, words = self.net.dec.free_running(
        #     code_t, code_w, max(batch.len))

        # Register hook
        code.register_hook(self.hooker.save_grad_norm)
        # code.register_hook(self.net.enc.save_ae_grad_norm_hook)
        # decoded.embed.register_hook(self.net.dec.save_ae_grad_norm_hook)

        # Compute word prediction loss and accuracy
        loss, acc = self._compute_recontruction_loss_acc(
            decoded.prob, batch.tar, len(self.net.vocab_w))

        loss.backward()

        # to prevent exploding gradient in RNNs
        self.net.embed_w.clip_grad_norm()
        self.net.enc.clip_grad_norm()
        self.net.dec.clip_grad_norm()

        # optimize
        self.net.optim_embed_w.step()
        self.net.optim_enc.step()
        self.net.optim_reg_ae.step()
        self.net.optim_dec.step()

        # NOTE: new!
        self.net.set_modules_train_mode(True)

        # real
        code_real = Variable(code.data, requires_grad=False)
        code_real_re = self.net.enc(decoded.embed.detach())
        code_real_sim = F.cosine_similarity(code_real, code_real_re).mean()
        code_real_sim.backward(retain_graph=True)

        code_real_re.register_hook(self.hooker.scale_grad_norm)
        self.net.disc.clamp_weights()
        disc_real = self.net.disc(code_real_re)
        disc_real.backward(self.pos_one)

        # fake
        code_fake = self.net.gen.for_train()
        decoded = self.net.dec.free_running(code_fake, self.cfg.max_len)
        code_fake_re = self.net.enc(decoded.embed.detach())
        code_fake_sim = F.cosine_similarity(code_fake, code_fake_re).mean()
        code_fake_sim.backward(retain_graph=True)

        code_fake_re.register_hook(self.hooker.scale_grad_norm)
        self.net.disc.clamp_weights()
        disc_fake = self.net.disc(code_fake_re)
        disc_fake.backward(self.neg_one)

        self.net.enc.clip_grad_norm()
        self.net.optim_enc.step()

        self.result.add(name, odict(
            loss=loss.data[0],
            acc=acc.data[0],
            # cosim=cos_sim.data[0],
            # var=self.net.reg.var,
            noise=self.net.enc.noise_radius,
        ))

    def _compute_recontruction_loss_acc(self, prob, target, vocab_size):
        masked_output, masked_target = \
            mask_output_target(prob, target, vocab_size)
        loss = self.net.dec.criterion_nll(masked_output, masked_target)
        _, max_ids = torch.max(masked_output, 1)
        acc = torch.mean(max_ids.eq(masked_target).float())

        return loss, acc

    def _eval_autoencoder(self, batch, decode_mode, name='AE_eval'):
        name += ('/' + decode_mode)
        self.net.set_modules_train_mode(False)

        # Build graph
        embed = self.net.embed_w(batch.src)
        code = self.net.enc(embed, batch.len)
        #code_var = self.net.reg.with_var(code)
        #cos_sim = F.cosine_similarity(code, code_var, dim=1).mean()

        if decode_mode == 'tf':
            decoded = self.net.dec.teacher_forcing(code, batch)
        elif decode_mode == 'fr':
            decoded = self.net.dec.free_running(code, max(batch.len))
        else:
            raise Exception("Unknown decode_mode type!")

        # Compute word prediction loss and accuracy
        loss, acc = self._compute_recontruction_loss_acc(
            decoded.prob, batch.tar, len(self.net.vocab_w))

        code_embed = ResultWriter.Embedding(
            embed=code.data,
            text=decoded.get_text_batch(),
            tag='code_embed')

        self.result.add(name, odict(
            loss=loss.data[0],
            acc=acc.data[0],
            embed=code_embed,
            # cosim=cos_sim.data[0],
            # var=self.net.reg.var,
            noise=self.net.enc.noise_radius,
            text=decoded.get_text_with_pair(batch.src),
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

    def _train_generator(self, name="Gen_train"):
        self.net.set_modules_train_mode(True)

        # Build graph
        code_fake = self.net.gen.for_train()
        disc_fake = self.net.disc(code_fake)

        # loss / backprop
        disc_fake.backward(self.pos_one)
        self.net.optim_gen.step()

        # NOTE : new!
        self.net.set_modules_train_mode(True)
        code_fake = Variable(code_fake.data, requires_grad=False)
        decoded_fake = self.net.dec.free_running(code_fake, self.cfg.max_len)
        code_fake_r = self.net.enc(decoded_fake.embed)
        disc_fake = self.net.disc(code_fake_r)
        disc_fake.backward(self.pos_one)

        self.net.dec.clip_grad_norm()
        self.net.optim_dec.step()

        self.result.add(name, odict(
            loss=disc_fake.data[0],
        ))

    def _train_discriminator(self, batch, name="Disc_train"):
        self.net.set_modules_train_mode(True)

        # Code generation
        embed = self.net.embed_w(batch.src)
        code_real = self.net.enc(embed, batch.len)
        #code_real = self.net.reg.with_var(code)
        code_fake = self.net.gen.for_train()

        # Grad hook : gradient scaling
        code_real.register_hook(self.hooker.scale_grad_norm)

        code_real = Variable(code_real.data, requires_grad=False)
        code_fake = Variable(code_fake.data, requires_grad=False)

        self.net.disc.clamp_weights()  # Weight clamping for WGAN
        disc_real = self.net.disc(code_real)
        disc_fake = self.net.disc(code_fake)
        loss_total = disc_real - disc_fake

        # WGAN backward
        disc_real.backward(self.pos_one)
        disc_fake.backward(self.neg_one)
        # loss_total.backward()
        self.net.optim_disc.step()

        # Gradient clipping
        # self.net.embed_w.clip_grad_norm()
        # self.net.enc.clip_grad_norm()
        #
        # self.net.optim_embed_w.step()
        # self.net.optim_enc.step()
        # self.net.optim_reg_ae.step()


        self.result.add(name, odict(
            loss_toal=loss_total.data[0],
            loss_real=disc_real.data[0],
            loss_fake=disc_fake.data[0],
        ))

    def _generate_text(self, name="Generated"):
        self.net.set_modules_train_mode(True)

        # Build graph
        code_fake = self.net.gen.for_eval()
        decoded = self.net.dec.tester.free_running(code_fake, self.cfg.max_len)

        code_embed = ResultWriter.Embedding(
            embed=code_fake.data,
            text=decoded.get_text_batch(),
            tag='code_embed')

        self.result.add(name, odict(
            embed=code_embed,
            txt_word=decoded.get_text(),
        ))

        # Evaluation
        scores = evaluate_sents(self.test_sents, decoded.get_text())
        self.result.add("Evaluation", scores)
