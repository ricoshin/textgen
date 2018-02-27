import logging
import math
import collections

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

from train.train_helper import ResultPackage, compute_cosine_sim
from utils.utils import to_gpu

log = logging.getLogger('main')
dict = collections.OrderedDict


def train_ae(cfg, net, batch, decoded):

    def mask_output_target(output, target, ntokens):
        # Create sentence length mask over padding
        target_mask = target.gt(0) # greater than 0
        masked_target = target.masked_select(target_mask)
        # target_mask.size(0) = batch_size*max_len
        # output_mask.size() : batch_size*max_len x ntokens
        target_mask = target_mask.unsqueeze(1)
        output_mask = target_mask.expand(target_mask.size(0), ntokens)
        # flattened_output.size(): batch_size*max_len x ntokens
        flattened_output = output.view(-1, ntokens)
        # flattened_output.masked_select(output_mask).size()
        #  num_of_masked_words(in batch, excluding <pad>)*ntokens
        masked_output = flattened_output.masked_select(output_mask)
        masked_output = masked_output.view(-1, ntokens)
        # masked_output.size() : num_of_masked_words x ntokens
        # masked_target : num_of_masked_words
        return masked_output, masked_target

    # compute word prediction loss and accuracy
    masked_output, masked_target = \
        mask_output_target(decoded.prob, batch.tar, cfg.vocab_size)
    loss_word = net.dec.criterion_nll(masked_output, masked_target)
    _, max_ids = torch.max(masked_output, 1)
    acc_word = torch.mean(max_ids.eq(masked_target).float())

    # compute tag prediction loss and accuracy
    # masked_output, masked_target = \
    #     mask_output_target(decoded.tag, batch.tar_tag, cfg.tag_size)
    # loss_tag = net.dec.criterion_ce(masked_output, masked_target)
    # _, max_ids = torch.max(masked_output, 1)
    # acc_tag = torch.mean(max_ids.eq(masked_target).float())

    loss_word.backward(retain_graph=True)
    # loss_tag.backward()
    loss_total = loss_word #+ loss_tag
    #loss_total.backward()

    # `clip_grad_norm` to prevent exploding gradient in RNNs / LSTMs
    torch.nn.utils.clip_grad_norm(net.enc.parameters(), cfg.clip)
    torch.nn.utils.clip_grad_norm(net.dec.parameters(), cfg.clip)

    # optimize
    net.optim_embed.step()
    net.optim_enc.step()
    net.optim_reg.step()
    # net.optim_mu.step()
    # net.optim_logvar.step()
    net.optim_dec.step()

    return ResultPackage('Autoencoder', dict(Loss_word=loss_word.data,
                                             #Loss_tag=loss_tag.data,
                                             Acc_word=acc_word.data[0]))
                                             #Acc_tag=acc_tag.data[0]))


def train_exposure(cfg, net, batch):
    net.embed.eval()
    net.enc.eval()
    net.dec.train()
    net.dec.zero_grad()

    # encode
    in_embed = net.embed(batch.src)
    code = net.enc(in_embed, noise=False, save_grad_norm=False)

    # decode
    code_new = Variable(code.data, requires_grad=False)
    embed_tf, _, _, _ = net.dec(code_new, batch.src, batch.len, mode='tf')
    embed_fr, _, _, _ = net.dec(code_new, max(batch.len), mode='fr')

    # encode again
    code_tf = net.enc(embed_tf, noise=False, save_grad_norm=False)
    code_fr = net.enc(embed_fr, noise=False, save_grad_norm=False)

    code_tar = Variable(code_tf.data, requires_grad=False)
    # [bsz, hidden_size]
    bsz = code_tar.size(0)
    # trick for batch-wise dot product
    similarity = torch.bmm(code_tar.view(bsz, 1, -1), code_fr.view(bsz, -1, 1))
    loss = torch.mean(similarity)
    #loss = net.enc.criterion_mse(code_fr, code_tar)
    loss.backward()


def eval_ae_tf(net, batch):
    # output.size(): batch_size x max_len x ntokens (logits)
    #output = ae(batch.src, batch.len, noise=True)
    embed = net.embed.tester(batch.src)
    code = net.enc.tester(embed)
    code_var = net.reg.tester.with_var(code)
    decoded = net.dec.tester.teacher_forcing(code_var, batch) #NOTE

    target = batch.tar.view(decoded.id.size(0), -1)
    outputs = decoded.id.data.cpu().numpy()
    targets = target.data.cpu().numpy()

    return targets, outputs


def eval_ae_fr(net, batch):
    # output.size(): batch_size x max_len x ntokens (logits)
    #code = ae.encode_only(cfg, batch, train=False)
    #max_ids, outs = ae.decode_only(cfg, code, vocab, train=False)
    embed = net.embed.tester(batch.src)
    code = net.enc.tester(embed)
    code_var = net.reg.tester.with_var(code)
    decoded = net.dec.tester.free_running(code_var, max(batch.len))

    target = batch.tar.view(decoded.id.size(0), -1)
    outputs = decoded.id.data.cpu().numpy()
    targets = target.data.cpu().numpy()

    return targets, outputs


def eval_gen_dec(cfg, net, fixed_noise):
    code_fake = net.gen.tester(fixed_noise)
    decoded = net.dec.tester.free_running(code_fake, cfg.max_len)
    outputs = decoded.id.data.cpu().numpy()

    return outputs


def recon_code_fake(cfg, net, code_fake):
    net.enc.eval()
    net.dec.eval()

    code_fake = to_gpu(cfg.cuda, Variable(code_fake.data, requires_grad=False))
    embed_fake, _, _, _ = net.dec(code_fake, cfg.max_len, mode='fr')
    code_fake_r = net.enc(embed_fake, noise=False)

    # register hook on logits of decoder
    def grad_hook(grad):
        if cfg.ae_grad_norm:
            gan_norm = torch.norm(grad, 2, 1).detach().data.mean()
            if gan_norm == .0:
                log.warning("zero sample_gan norm!")
                normed_grad = grad
            else:
                normed_grad = grad * net.dec.grad_norm / gan_norm
        else:
            normed_grad = grad

        # normed_grad *= math.fabs(cfg.gan_to_ae)
        return normed_grad

    embed_fake.register_hook(grad_hook)

    return code_fake_r


def train_gen_s(cfg, net, code_fake, vocab):
    net.enc.eval()
    net.dec.train()
    net.dec.zero_grad()
    # net.disc_c.eval()

    code_fake = to_gpu(cfg.cuda, Variable(code_fake.data, requires_grad=False))
    embed_fake, _, _, _ = net.dec(code_fake, cfg.max_len, mode='fr')
    code_fake_r = net.enc(embed_fake, noise=False)

    # register hook on logits of decoder
    def grad_hook(grad):
        if cfg.ae_grad_norm:
            gan_norm = torch.norm(grad, 2, 1).detach().data.mean()
            if gan_norm == .0:
                log.warning("zero sample_gan norm!")
                normed_grad = grad
            else:
                normed_grad = grad * net.dec.grad_norm / gan_norm
        else:
            normed_grad = grad

        # normed_grad *= math.fabs(cfg.gan_to_ae)
        return normed_grad

    embed_fake.register_hook(grad_hook)

    # loss
    # gan_loss, pred = net.disc_c(code_fake_r)
    # one = to_gpu(cfg.cuda, torch.FloatTensor([1]))
    # gan_loss.backward(one, retain_graph=True)
    # pred_mean = pred.mean()

    reg_loss = net.enc.criterion_mse(code_fake_r, code_fake)
    reg_loss.backward()

    #loss = net.enc.criterion_mse(code_fr, code_tar)

    # `clip_grad_norm` to prevent exploding gradient in RNNs / LSTMs
    torch.nn.utils.clip_grad_norm(net.dec.parameters(), cfg.clip)

    return ResultPackage("Decoder_Loss",
                         dict(loss=reg_loss.data[0]))


def train_enc(cfg, net, disc_real):
    # loss / backprop
    one = to_gpu(cfg.cuda, torch.FloatTensor([1]))
    disc_real.backward(one * -1, retain_graph=True)

    result = ResultPackage("Generator_Loss",
                           dict(loss=disc_real.data[0]))

    net.optim_logvar.step()

    return result


def train_gen(cfg, net, disc_fake):
    # loss / backprop
    one = to_gpu(cfg.cuda, torch.FloatTensor([1]))
    disc_fake.backward(one, retain_graph=True) # Note reuse fake code in train_gen_s

    result = ResultPackage("Generator_Loss",
                           dict(loss=disc_fake.data[0]))

    net.optim_gen_c.step()

    return result


def generate_codes(cfg, net, batch):
    net.enc.train() # NOTE train encoder!
    net.enc.zero_grad()
    net.gen.eval()

    in_embed = net.embed(batch.src)
    code_real = net.enc(in_embed, noise=False)
    code_fake = net.gen(None)

    return code_real, code_fake


def train_disc_c(cfg, net, disc_real, disc_fake):
    # WGAN backward
    one = to_gpu(cfg.cuda, torch.FloatTensor([1]))
    disc_real.backward(one, retain_graph=True)
    disc_fake.backward(one * -1, retain_graph=True)
    loss_total = disc_real - disc_fake

    # Prediction layer (for interpretation)
    # label_real = to_gpu(cfg.cuda, Variable(torch.ones(pred_real.size())))
    # label_fake = to_gpu(cfg.cuda, Variable(torch.zeros(pred_fake.size())))
    # loss_pred_real = net.disc_s.criterion_bce(pred_real, label_real)
    # loss_pred_fake = net.disc_s.criterion_bce(pred_fake, label_fake)

    # pred mean
    # pred_real_mean = pred_real.mean()
    # pred_fake_mean = pred_fake.mean()

    # backprop.
    # loss_pred_real.backward()
    # loss_pred_fake.backward()

    torch.nn.utils.clip_grad_norm(net.enc.parameters(), cfg.clip)

    net.optim_enc.step()
    net.optim_mu.step()
    net.optim_disc_c.step()

    return ResultPackage("Code_GAN_Loss",
                         dict(D_Loss_Total=loss_total.data[0],
                              D_Loss_Real=disc_real.data[0],
                              D_Loss_Fake=disc_fake.data[0]))
                              # D_Pred_Real=pred_real_mean.data[0],
                              # D_Pred_Fake=pred_fake_mean.data[0]))


def train_disc_s(cfg, net, batch, code_real, code_fake):
    pass
    # net.dec.eval()
    # net.disc_s.train()
    # net.disc_s.zero_grad()
    #
    # ids_real, _, outs_real = net.dec.generate(code_real)
    # ids_fake, _, outs_fake = net.dec.generate(code_fake)
    #
    # # "real" fake (embeddings)
    # outs_fake = torch.cat([outs_real, outs_fake], dim=0)
    # code_fake = torch.cat([code_real, code_fake], dim=0)
    #
    # # clamp parameters to a cube
    # for p in net.disc_s.parameters():
    #     p.data.clamp_(-cfg.gan_clamp, cfg.gan_clamp) # [min,max] clamp
    #     # WGAN clamp (default:0.01)
    #
    # pred_real, attn_real = net.disc_s(batch.src.detach())
    # pred_fake, attn_fake = net.disc_s(outs_fake.detach())
    #
    # # GAN loss
    # label_real = to_gpu(cfg.cuda, Variable(torch.ones(pred_real.size())))
    # label_fake = to_gpu(cfg.cuda, Variable(torch.zeros(pred_fake.size())))
    # loss_real = net.disc_s.criterion_bce(pred_real, label_real)
    # loss_fake = net.disc_s.criterion_bce(pred_fake, label_fake)
    # loss_total = loss_real + loss_fake
    #
    # # pred mean
    # real_mean = pred_real.mean()
    # fake_mean = pred_fake.mean()
    #
    # # backprop.
    # loss_real.backward()
    # loss_fake.backward()
    #
    # # results
    # loss_gan = ResultPackage("Sample_GAN_loss",
    #                          dict(D_Total=loss_total,
    #                               D_Real=loss_real,
    #                               D_Fake=loss_fake))
    # pred_gan = ResultPackage("Sample_GAN_pred",
    #                          dict(D_real=real_mean,
    #                               D_Fake=fake_mean))
    #
    # ids = [batch.src.data.cpu().numpy(), ids_fake]
    # attns = [attn_real, attn_fake]
    #
    # return loss_gan, pred_gan, ids, attns
