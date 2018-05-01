import logging
import os
import time
from collections import OrderedDict
from test.evaluate import evaluate_sents

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.decoder import DecoderRNN
from torch.autograd import Variable
from train.supervisor import TrainingSupervisor
from train.train_helper import (GradientScalingHook, GradientTransferHook,
                                load_test_data, mask_output_target, SigmaHook)
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

        self.code_var_hook = GradientScalingHook()
        #self.code_var_hook = GradientScalingHook()
        #self.tansfer_hook = GradientTransferHook()

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

                # train generator(with disc) / decoder(with disc_s)
                for i in range(cfg.niter_gan_g):  # default: 1
                    self._train_generator()

            #self._train_regularizer2(batch)

        if sv.is_evaluation():
            with sv.evaluation_context():
                batch = net.data_eval.next()
                #self._eval_autoencoder(batch, 'tf')
                self._eval_autoencoder(batch)
                self._generate_text()

    def _train_autoencoder(self, batch, name='AE_train'):
        self.net.set_modules_train_mode(True)

        # Build graph
        embed = self.net.embed_w(batch.src)
        enc_h = self.net.enc(embed, batch.len)
        code_var = self.net.reg.with_var(enc_h)
        #self.sigma_hook.save_sign(self.net.reg.std)
        #cos_sim = F.cosine_similarity(code, code_var, dim=1).mean()
        decoded = self.net.dec(code_var, batch=batch)
        # tags, words = self.net.dec.free_running(
        #     code_t, code_w, max(batch.len))

        # Register hook
        #code.register_hook(self.code_hook.save_grad_norm)
        code_var.register_hook(self.code_var_hook.save_grad_norm)
        # code.register_hook(self.net.enc.save_ae_grad_norm_hook)
        # decoded.embed.register_hook(self.net.dec.save_ae_grad_norm_hook)

        # Compute word prediction loss and accuracy
        loss_recon, acc = self._compute_recon_loss_and_acc(
            decoded.prob, batch.tar, len(self.net.vocab_w))
        #loss_var = 1 / torch.sum(self.net.reg.var) * 0.0000001
        #loss_mean = code_var.mean()
        #loss_var = loss_recon.detach() / loss_var.detach() * loss_var * 0.2
        #loss_kl = self._compute_kl_div_loss(mu, sigma)
        loss = loss_recon# + loss_var

        loss.backward()

        # to prevent exploding gradient in RNNs
        self.net.embed_w.clip_grad_norm()
        self.net.enc.clip_grad_norm()
        self.net.reg.clip_grad_norm()
        self.net.dec.clip_grad_norm()

        # optimize
        self.net.optim_embed_w.step()
        self.net.optim_enc.step()
        self.net.optim_reg_mu.step()
        self.net.optim_reg_sigma_gen.step()
        self.net.optim_dec.step()

        self.result.add(name, odict(
            loss_total=loss.data[0],
            loss_recon=loss_recon.data[0],
            #loss_var=loss_var.data[0],
            acc=acc.data[0],
            sigma=self.net.reg.sigma,
            # cosim=cos_sim.data[0],
            # var=self.net.reg.var,
            noise=self.net.enc.noise_radius,
        ))

    def _eval_autoencoder(self, batch, name='AE_eval'):
        #name += ('/' + decode_mode)
        n_vars = 5
        assert n_vars > 0
        codes_var = list()

        self.net.set_modules_train_mode(False)

        # Build graph
        embed = self.net.embed_w(batch.src)
        enc_h = self.net.enc(embed, batch.len)
        code = self.net.reg.without_var(enc_h)
        for _ in range(n_vars):
            code_var = self.net.reg.with_var(enc_h)
            codes_var.append(code_var)

        #code_var = self.net.reg.with_var(code)
        #cos_sim = F.cosine_similarity(code, code_var, dim=1).mean()
        assert len(codes_var) > 0
        decoded = self.net.dec(code_var, max_len=batch.max_len)

        # Compute word prediction loss and accuracy
        loss_recon, acc = self._compute_recon_loss_and_acc(
            decoded.prob, batch.tar, len(self.net.vocab_w))
        #loss_var = 1 / torch.mean(self.net.reg.var)
        #loss_kl = self._compute_kl_div_loss(mu, sigma)

        embed = ResultWriter.Embedding(
            embed=code.data,
            text=decoded.get_text_batch(),
            tag='code_embed')

        embeds_var = odict()
        for i in range(n_vars):
            embed_var = ResultWriter.Embedding(
                embed=codes_var[i].data,
                text=decoded.get_text_batch(),
                tag='code_embed')
            embeds_var.update({('embed_var_%d' % i): embed_var})

        result_dict = odict(
            loss_recon=loss_recon.data[0],
            #loss_var=loss_var.data[0],
            #loss_kl=loss_kl.data[0],
            acc=acc.data[0],
            embed=embed,
            #embed_var=embed_var,
            # cosim=cos_sim.data[0],
            sigma=self.net.reg.sigma,
            noise=self.net.enc.noise_radius,
            text=decoded.get_text_with_pair(batch.src),
        )
        result_dict.update(embeds_var)
        self.result.add(name, result_dict)

    def _compute_recon_loss_and_acc(self, output, target, vocab_size):
        output = output.view(-1, vocab_size)  # flatten output
        if self.cfg.dec_type == 'rnn':
            output, target = mask_output_target(output, target, vocab_size)
        loss = self.net.dec.criterion_nll(output, target)
        _, max_ids = torch.max(output, 1)
        acc = torch.mean(max_ids.eq(target).float())

        return loss, acc

    def _compute_kl_div_loss(self, mean, stddev):
        mean_sq = mean * mean
        stddev_sq = stddev * stddev
        return 0.5 * torch.mean(mean_sq + stddev_sq - torch.log(stddev_sq) - 1)

    def _train_regularizer(self, batch, name="Logvar_train"):
        self.net.set_modules_train_mode(True)

        # Build graph
        embed = self.net.embed_w(batch.src)
        code_enc = self.net.enc(embed, batch.len)
        #code_var = self.net.reg.with_var(code)
        noise = self.net.rev(code_enc)
        code_gen = self.net.gen(noise)
        code_diff = code_gen - code_enc

        rev_dist = F.pairwise_distance(code_enc, code_gen, p=2).mean()
        #code_enc.register_hook(self.tansfer_hook.stash_grad)
        rev_dist.backward(retain_graph=True)

        code_enc_var = self.net.reg.with_directional_var(code_enc, code_diff)
        rev_dist = F.pairwise_distance(code_enc_var, code_gen, p=2).mean()
        #code_enc_var.register_hook(self.tansfer_hook.transfer_grad)
        rev_dist.backward(retain_graph=True)

        # loss / backprop
        #disc_real.backward(self.neg_one)
        self.net.optim_reg_ae.step()

        self.net.set_modules_train_mode(True)
        disc_real = self.net.disc(code_enc)
        disc_real.backward(self.neg_one)
        #self.net.optim_embed_w.step()
        self.net.optim_enc.step()

        self.result.add(name, odict(loss=rev_dist.data[0]))

    def _train_generator(self, name="Gen_train"):
        self.net.set_modules_train_mode(True)

        # Build graph
        noise = self.net.gen.get_noise()
        code_fake = self.net.gen(noise)
        self.net.disc.clamp_weights()
        disc_fake = self.net.disc(code_fake)

        # loss / backprop
        disc_fake.backward(self.pos_one)
        self.net.optim_gen.step()

        # noise_recon = self.net.rev(code_fake.detach())
        # rev_dist = F.pairwise_distance(noise, noise_recon, p=2).mean()
        # rev_dist.backward()
        # self.net.optim_rev.step()

        self.result.add(name, odict(
            loss_gen=disc_fake.data[0],
            #loss_rev=rev_dist.data[0],
        ))

    def _train_regularizer2(self, batch, name="Reg_train"):
        self.net.set_modules_train_mode(True)

        embed = self.net.embed_w(batch.src)
        enc_h = self.net.enc(embed, batch.len)
        code_var = self.net.reg.with_var(enc_h)
        self.net.disc.clamp_weights()
        disc_var = self.net.disc(code_var)

        #code_var.register_hook(self.code_var_hook.scale_grad_norm)
        disc_var.backward(self.pos_one)
        #self.net.embed_w.clip_grad_norm()
        #self.net.enc.clip_grad_norm()
        #self.net.reg.clip_grad_norm()
        self.net.optim_embed_w.step()
        self.net.optim_enc.step()
        self.net.optim_reg_sigma_gen.step()

    def _train_discriminator(self, batch, name="Disc_train"):
        self.net.set_modules_train_mode(True)

        # Code generation
        embed = self.net.embed_w(batch.src)
        enc_h = self.net.enc(embed, batch.len)
        code_real = self.net.reg.without_var(enc_h)
        code_fake = self.net.gen.for_train()
        #self.net.reg.sigma.register_hook(lambda grad: grad*grad.lt(0).float())

        # Grad hook : gradient scaling
        #code_real.register_hook(self.code_hook.scale_grad_norm)
        #code_posvar.register_hook(self.hook.scale_grad_norm)
        #code_negvar.register_hook(self.hook.scale_grad_norm)

        self.net.disc.clamp_weights()  # Weight clamping for WGAN
        disc_real = self.net.disc(code_real.detach())
        #disc_real_neg = self.net.disc(code_negvar.detach())
        #disc_real_neg = self.net.disc(code_neg)
        disc_fake = self.net.disc(code_fake.detach())
        loss_total = disc_real - disc_fake

        #code_var.register_hook(self.hook_pos.stash_abs_grad)
        #code_neg.register_hook(self.hook_pos.pass_smaller_abs_grad)

        # WGAN backward
        disc_real.backward(self.pos_one)
        disc_fake.backward(self.neg_one)
        # loss_total.backward()
        #self.net.optim_reg_ae.step()
        self.net.optim_disc.step()

        # train encoder adversarilly
        # self.net.embed_w.zero_grad()
        # self.net.enc.zero_grad()
        # self.net.reg.zero_grad()
        # disc_real.backward(self.neg_one)
        # self.net.embed_w.clip_grad_norm()
        # self.net.enc.clip_grad_norm()
        # self.net.optim_embed_w.step()
        # self.net.optim_enc.step()
        # self.net.optim_reg_mu.step()

        self.result.add(name, odict(
            loss_toal=loss_total.data[0],
            loss_real=disc_real.data[0],
            loss_fake=disc_fake.data[0],
        ))


    def _generate_text2(self, name="Generated"):
        self.net.set_modules_train_mode(True)

        # Build graph
        noise_size = (self.cfg.eval_size, self.cfg.hidden_size_w)
        code_fake = self.net.dec.get_noise(noise_size)
        decoded = self.net.dec.tester(code_fake, max_len=self.cfg.max_len)

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



    def _generate_text(self, name="Generated"):
        self.net.set_modules_train_mode(True)

        # Build graph
        code_fake = self.net.gen.for_eval()
        decoded = self.net.dec.tester(code_fake, max_len=self.cfg.max_len)

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
