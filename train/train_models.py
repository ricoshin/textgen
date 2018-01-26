import math
import collections

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

from train.train_helper import ResultPackage
from utils.utils import to_gpu

dict = collections.OrderedDict


def train_ae(cfg, net, batch, optimize=True):
    # forward
    code = net.enc(batch.src, batch.len, ae_mode=True, train=True)
    output = net.dec(code, batch.src, batch.len, ae_mode=True, train=True)
    # output.size(): batch_size x max_len x ntokens (logits)

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

    # loss & accuracy
    masked_output, masked_target = \
        mask_output_target(output, batch.tar, cfg.vocab_size)

    max_vals, max_indices = torch.max(masked_output, 1)
    accuracy = torch.mean(max_indices.eq(masked_target).float())
    loss = net.dec.criterion_ce(masked_output, masked_target)

    # backward
    loss.backward(retain_graph=(not optimize))

    # gradient clipping
    torch.nn.utils.clip_grad_norm(net.enc.parameters(), cfg.clip)
    torch.nn.utils.clip_grad_norm(net.dec.parameters(), cfg.clip)

    # optimize
    if optimize:
        net.optim_enc.step()
        net.optim_dec.step()

    return ResultPackage("Autoencoder",
                         dict(Loss=loss.data, Accuracy=accuracy.data[0]))


def eval_ae_tf(cfg, net, batch):
    # forward / NOTE : ae_mode off?
    code = net.enc(batch.src, batch.len, ae_mode=False, train=False)
    output = net.dec(code, batch.src, batch.len, ae_mode=True, train=False)
    # output.size(): batch_size x max_len x ntokens (logits)

    max_value, max_indices = torch.max(output, 2)
    target = batch.tar.view(output.size(0), -1)
    outputs = max_indices.data.cpu().numpy()
    targets = target.data.cpu().numpy()

    return targets, outputs

def eval_ae_fr(cfg, net, batch):
    # forward / NOTE : ae_mode off?
    # "real" real
    code = net.enc(batch.src, batch.len, ae_mode=False, train=False)
    max_ids, outputs = net.dec(code, ae_mode=False, train=False)
    # output.size(): batch_size x max_len x ntokens (logits)
    target = batch.tar.view(outputs.size(0), -1)
    targets = target.data.cpu().numpy()

    return targets, max_ids


def train_dec(cfg, net, fake_code, optimize=True):
    # forward
    fake_code = fake_code.detach() # NOTE cut the graph
    fake_ids, fake_outs = net.dec(fake_code, train=True)
    # fake_outs.size() : [batch_size*2, max_len, vocab_size]
    _, pred_fake, attn_fake = net.disc_s(fake_outs, train=False)

    # register hook on logits of decoder
    def grad_hook(grad):
        if cfg.ae_grad_norm:
            gan_norm = torch.norm(grad, 2, 1).detach().data.mean()
            if gan_norm == .0:
                log.warning("zero sample_gan norm!")
                import ipdb; ipdb.set_trace()
                normed_grad = grad
            else:
                normed_grad = grad * net.dec.grad_norm / gan_norm
        else:
            normed_grad = grad

        normed_grad *= math.fabs(cfg.gan_to_ae)
        return normed_grad

    net.dec.logits.register_hook(grad_hook)

    # loss
    label_real = to_gpu(cfg.cuda, Variable(torch.ones(pred_fake.size())))
    loss = net.disc_s.criterion_bce(pred_fake, label_real)

    # pred average
    mean = pred_fake.mean()

    # backward
    loss.backward(retain_graph=(not optimize))

    # optimize
    if optimize:
        net.optim_dec.step()

    return ResultPackage("Decoder_Loss",
                         dict(loss=loss.data[0], pred=mean.data[0]))


def train_gen(cfg, net, optimize=True):
    # forward
    fake_code = net.gen(train=True)
    err_g = net.disc_c(fake_code, train=False)

    # backward
    one = to_gpu(cfg.cuda, torch.FloatTensor([1]))
    err_g.backward(one, retain_graph=(not optimize))

    # optimize
    if optimize:
        net.optim_gen.step()

    # result
    result = ResultPackage("Generator_Loss", dict(loss=err_g.data[0]))

    return result, fake_code


def train_disc_c(cfg, net, code_real, code_fake, optimize=True):
    # NOTE : encoder will be trained adversarially with disc_c
    # clamp parameters to a cube
    for p in net.disc_c.parameters():
        p.data.clamp_(-cfg.gan_clamp, cfg.gan_clamp) # [min,max] clamp
        # WGAN clamp (default:0.01)

    # positive samples ----------------------------
    def grad_hook(grad):
        # Gradient norm: regularize to be same
        # code_grad_gan * code_grad_ae / norm(code_grad_gan)

        # regularize GAN gradient in AE(encoder only) gradient scale
        # GAN gradient * [norm(Encoder gradient) / norm(GAN gradient)]
        if cfg.ae_grad_norm: # norm code gradient from critic->encoder
            gan_norm = torch.norm(grad, 2, 1).detach().data.mean()
            if gan_norm == .0:
                log.warning("zero code_gan norm!")
                import ipdb; ipdb.set_trace()
                normed_grad = grad
            else:
                normed_grad = grad * net.enc.grad_norm / gan_norm
            # grad : gradient from GAN
            # aeoder.grad_norm : norm(gradient from AE)
            # gan_norm : norm(gradient from GAN)
        else:
            normed_grad = grad

        # weight factor and sign flip (adversarial training)
        normed_grad *= -math.fabs(cfg.gan_to_ae)

        return normed_grad

    code_real.register_hook(grad_hook) # normed_grad

    # forward
    err_d_real = net.disc_c(code_real.detach(), train=True) # NOTE:train encoder as well
    err_d_fake = net.disc_c(code_fake.detach(), train=True) # NOTE:dectach gen
    err_d = -(err_d_real - err_d_fake)

    # backward
    one = to_gpu(cfg.cuda, torch.FloatTensor([1]))
    err_d_real.backward(one, retain_graph=(not optimize))
    err_d_fake.backward((-1*one), retain_graph=(not optimize))

    # clip encoder's gradient
    torch.nn.utils.clip_grad_norm(net.enc.parameters(), cfg.clip)
    # optimize
    if optimize:
        net.optim_enc.step()
        net.optim_disc_c.step()

    return ResultPackage("Code_GAN_Loss",
                         dict(D_Total=err_d.data[0],
                              D_Real=err_d_real.data[0],
                              D_Fake=err_d_fake.data[0]))


def train_disc_s(cfg, net, in_real, in_fake, code_real, code_fake,
                 optimize=True):
    # clamp parameters to a cube
    for p in net.disc_s.parameters():
        p.data.clamp_(-cfg.gan_clamp, cfg.gan_clamp) # [min,max] clamp
        # WGAN clamp (default:0.01)

    # forward
    rec_real, pred_real, attn_real = net.disc_s(in_real.detach(), train=True)
    rec_fake, pred_fake, attn_fake = net.disc_s(in_fake.detach(), train=True)

    # attention loss
    label_real = to_gpu(cfg.cuda, Variable(torch.ones(pred_real.size())))
    label_fake = to_gpu(cfg.cuda, Variable(torch.zeros(pred_fake.size())))
    l_gan_real = net.disc_s.criterion_bce(pred_real, label_real)
    l_gan_fake = net.disc_s.criterion_bce(pred_fake, label_fake)

    # reconstruction loss
    l_rec_real = F.cosine_similarity(rec_real.squeeze(), code_real.detach())
    l_rec_fake = F.cosine_similarity(rec_fake.squeeze(), code_fake.detach())
    l_rec_real = torch.sum(l_rec_real, dim=0)
    l_rec_fake = torch.sum(l_rec_fake, dim=0)

    # total loss
    l_gan_total = l_gan_real + l_gan_fake
    l_rec_total = l_rec_real + l_rec_fake
    l_total = l_gan_total + l_rec_total

    # pred mean
    real_mean = pred_real.mean()
    fake_mean = pred_fake.mean()

    # backward
    l_real = l_gan_real + l_rec_real
    l_fake = l_gan_fake + l_rec_fake
    l_real.backward(retain_graph=(not optimize))
    l_fake.backward(retain_graph=(not optimize))

    # optimize
    if optimize:
        net.optim_disc_s.step()

    # results
    loss_gan = ResultPackage("Sample_GAN_loss",
                             dict(D_Total=l_gan_total,
                                  D_Real=l_gan_real,
                                  D_Fake=l_gan_fake))
    loss_rec = ResultPackage("Sample_Recon_loss",
                             dict(D_Total=l_rec_total,
                                  D_Feal=l_rec_real,
                                  D_Fake=l_rec_fake))
    pred_gan = ResultPackage("Sample_GAN_pred",
                             dict(D_real=real_mean,
                                  D_Fake=fake_mean))
    attn = [attn_real, attn_fake]

    return loss_gan, loss_rec, pred_gan, attn
