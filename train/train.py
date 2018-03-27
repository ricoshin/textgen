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
from train.train_helper import (load_test_data, mask_output_target,
                                GradientScalingHooker)
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

        self.hooker_w = GradientScalingHooker(net.cfg.gan_to_enc)
        self.hooker_t = GradientScalingHooker(net.cfg.gan_to_enc)

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
            for k in range(sv.niter_gan): # epc0=1, epc2=2, epc4=3, epc6=4

                # train discriminator/critic (at a ratio of 5:1)
                for i in range(cfg.niter_gan_d): # default: 5
                    batch = net.data_gan.next()
                    self._train_discriminator(batch)
                    #self._train_regularizer(batch)

                # train generator(with disc_c) / decoder(with disc_s)
                for i in range(cfg.niter_gan_g): # default: 1
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
        code_t, code_w = self.net.enc(embed, batch.len)
        #code_var = self.net.reg.with_var(code)
        #cos_sim = F.cosine_similarity(code, code_var, dim=1).mean()
        tags, words = self.net.dec.teacher_forcing(code_t, code_w, batch)
        # tags, words = self.net.dec.free_running(
        #     code_t, code_w, max(batch.len))

        # Register hook
        code_t.register_hook(self.hooker_t.save_grad_norm)
        code_w.register_hook(self.hooker_w.save_grad_norm)
        #code.register_hook(self.net.enc.save_ae_grad_norm_hook)
        #decoded.embed.register_hook(self.net.dec.save_ae_grad_norm_hook)

        # Compute word prediction loss and accuracy
        loss_t, acc_t = self._compute_recontruction_loss_acc(
            tags.prob, batch.tar_tag, len(self.net.vocab_t))
        loss_w, acc_w = self._compute_recontruction_loss_acc(
            words.prob, batch.tar, len(self.net.vocab_w))

        loss_total = loss_t + loss_w
        loss_total.backward()

        # `clip_grad_norm` to prevent exploding gradient in RNNs / LSTMs
        self.net.embed_t.clip_grad_norm()
        self.net.embed_w.clip_grad_norm()
        self.net.enc.clip_grad_norm()
        self.net.dec.clip_grad_norm()

        # optimize
        self.net.optim_embed_t.step()
        self.net.optim_embed_w.step()
        self.net.optim_enc.step()
        self.net.optim_reg_ae.step()
        self.net.optim_dec.step()

        self.result.add(name, odict(
            loss_tag=loss_t.data[0],
            loss_word=loss_w.data[0],
            acc_tag=acc_t.data[0],
            acc_word=acc_w.data[0],
            #cosim=cos_sim.data[0],
            #var=self.net.reg.var,
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
        code_t, code_w = self.net.enc(embed, batch.len)
        #code_var = self.net.reg.with_var(code)
        #cos_sim = F.cosine_similarity(code, code_var, dim=1).mean()

        if decode_mode == 'tf':
            tags, words = self.net.dec.teacher_forcing(code_t, code_w, batch)
        elif decode_mode == 'fr':
            tags, words = self.net.dec.free_running(
                code_t, code_w, max(batch.len))
        else:
            raise Exception("Unknown decode_mode type!")

        # Compute word prediction loss and accuracy
        loss_t, acc_t = self._compute_recontruction_loss_acc(
            tags.prob, batch.tar_tag, len(self.net.vocab_t))
        loss_w, acc_w = self._compute_recontruction_loss_acc(
            words.prob, batch.tar, len(self.net.vocab_w))

        code_t_embed = ResultWriter.Embedding(
            embed=code_t.data,
            text=tags.get_text_batch(),
            tag='embed_tags')
        code_w_embed = ResultWriter.Embedding(
            embed=code_w.data,
            text=words.get_text_batch(),
            tag='embed_words')

        self.result.add(name, odict(
            loss_tag=loss_t.data[0],
            loss_word=loss_w.data[0],
            acc_tag=acc_t.data[0],
            acc_word=acc_w.data[0],
            embed_tag=code_t_embed,
            embed_word=code_w_embed,
            #cosim=cos_sim.data[0],
            #var=self.net.reg.var,
            noise=self.net.enc.noise_radius,
            txt_tags=tags.get_text_with_pair(batch.src_tag),
            txt_words=words.get_text_with_pair(batch.src),
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
        code_t_fake, code_w_fake = self.net.gen.for_train()
        disc_t_fake = self.net.disc_t(code_t_fake)
        disc_w_fake = self.net.disc_w(code_w_fake)


        # loss / backprop
        disc_t_fake.backward(self.pos_one, retain_graph=True)
        disc_w_fake.backward(self.pos_one)
        self.net.optim_gen.step()

        self.result.add(name, odict(
            loss_t=disc_t_fake.data[0],
            loss_w=disc_w_fake.data[0],
        ))

    def _train_discriminator(self, batch, name="Disc_train"):
        self.net.set_modules_train_mode(True)

        # Code generation
        embed = self.net.embed_w(batch.src)
        code_t_real, code_w_real = self.net.enc(embed, batch.len)
        #code_real = self.net.reg.with_var(code)
        code_t_fake, code_w_fake = self.net.gen.for_train()

        # Grad hook : gradient scaling
        code_t_real.register_hook(self.hooker_t.scale_grad_norm)
        code_w_real.register_hook(self.hooker_w.scale_grad_norm)

        # Weight clamping for WGAN
        self.net.disc_t.clamp_weights()
        self.net.disc_w.clamp_weights()

        disc_t_real = self.net.disc_t(code_t_real)
        disc_t_fake = self.net.disc_t(code_t_fake.detach())
        loss_t_total = disc_t_real - disc_t_fake

        disc_w_real = self.net.disc_w(code_w_real)
        disc_w_fake = self.net.disc_w(code_w_fake.detach())
        loss_w_total = disc_w_real - disc_w_fake

        # WGAN backward
        disc_t_real.backward(self.pos_one, retain_graph=True)
        disc_t_fake.backward(self.neg_one, retain_graph=True)
        disc_w_real.backward(self.pos_one)
        disc_w_fake.backward(self.neg_one)
        # loss_total.backward()

        # Gradient clipping
        self.net.embed_t.clip_grad_norm()
        self.net.embed_w.clip_grad_norm()
        self.net.enc.clip_grad_norm()
        self.net.dec.clip_grad_norm()

        self.net.optim_embed_t.step()  # NOTE
        self.net.optim_embed_w.step()
        self.net.optim_enc.step()
        #self.net.optim_reg_ae.step()
        self.net.optim_disc_t.step()
        self.net.optim_disc_w.step()

        self.result.add(name, odict(
            loss_t_toal=loss_t_total.data[0],
            loss_t_real=disc_t_real.data[0],
            loss_t_fake=disc_t_fake.data[0],
            loss_w_toal=loss_w_total.data[0],
            loss_w_real=disc_w_real.data[0],
            loss_w_fake=disc_w_fake.data[0],
        ))

    def _generate_text(self, name="Generated"):
        self.net.set_modules_train_mode(True)

        # Build graph
        code_t_fake, code_w_fake = self.net.gen.for_eval()
        tags, words = self.net.dec.free_running(
            code_t_fake, code_w_fake, self.cfg.max_len)

        code_t_embed = ResultWriter.Embedding(
            embed=code_t_fake.data,
            text=tags.get_text_batch(),
            tag='embed_tags')
        code_w_embed = ResultWriter.Embedding(
            embed=code_w_fake.data,
            text=words.get_text_batch(),
            tag='embed_words')

        self.result.add(name, odict(
            embed_tag=code_t_embed,
            embed_word=code_w_embed,
            txt_tag=tags.get_text(),
            txt_word=words.get_text(),
        ))

        # Evaluation
        scores = evaluate_sents(self.test_sents, words.get_text())
        self.result.add("Evaluation", scores)
