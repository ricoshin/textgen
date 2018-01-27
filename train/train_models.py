import math

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

from train.train_helper import ResultPackage
from utils.utils import to_gpu


def train_ae(cfg, ae, batch):
    ae.train()
    ae.zero_grad()

    # output.size(): batch_size x max_len x ntokens (logits)
    output = ae(batch.src, batch.len, noise=True)

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

    masked_output, masked_target = \
        mask_output_target(output, batch.tar, cfg.vocab_size)

    max_vals, max_indices = torch.max(masked_output, 1)
    accuracy = torch.mean(max_indices.eq(masked_target).float())

    loss = ae.criterion_ce(masked_output, masked_target)

    loss.backward()
    # `clip_grad_norm` to prevent exploding gradient in RNNs / LSTMs
    torch.nn.utils.clip_grad_norm(ae.parameters(), cfg.clip)
    return ResultPackage("Autoencoder",
                         dict(Loss=loss.data, Accuracy=accuracy.data[0]))


def eval_ae_tf(cfg, ae, batch):
    ae.eval()
    # output.size(): batch_size x max_len x ntokens (logits)
    output = ae(batch.src, batch.len, noise=True)

    max_value, max_indices = torch.max(output, 2)
    target = batch.tar.view(output.size(0), -1)
    outputs = max_indices.data.cpu().numpy()
    targets = target.data.cpu().numpy()

    return targets, outputs


def eval_ae_fr(cfg, ae, batch, vocab):
    ae.eval()
    # output.size(): batch_size x max_len x ntokens (logits)
    code = ae.encode_only(cfg, batch, train=False)
    max_ids, outs = ae.decode_only(cfg, code, vocab, train=False)

    targets = batch.tar.view(outs.size(0), -1)
    targets = targets.data.cpu().numpy()

    return targets, max_ids


def train_dec(cfg, ae, disc_s, fake_code, vocab):
    ae.train()
    ae.zero_grad()

    fake_ids, fake_outs = ae.decode_only(cfg, fake_code, vocab)
    # fake_outs.size() : [batch_size*2, max_len, vocab_size]

    if cfg.disc_s_in == 'embed':
        # fake_in.size() : [batch_size*2, max_len, vocab_size]
        # soft embedding lookup (3D x 2D matrix multiplication)
        fake_outs = disc_s.soft_embed(fake_outs)
        # [batch_size, max_len, embed_size]

    def grad_hook(grad):
        if cfg.ae_grad_norm:
            gan_norm = torch.norm(grad, 2, 1).detach().data.mean()
            if gan_norm == .0:
                log.warning("zero sample_gan norm!")
                import ipdb; ipdb.set_trace()
                normed_grad = grad
            else:
                normed_grad = grad * ae.dec_grad_norm / gan_norm
        else:
            normed_grad = grad

        normed_grad *= math.fabs(cfg.gan_toenc)
        return normed_grad

    fake_outs.register_hook(grad_hook)

    # loss
    _, pred_fake, attn_fake = disc_s(fake_outs)
    label_real = to_gpu(cfg.cuda, Variable(torch.ones(pred_fake.size())))
    loss = disc_s.criterion_bce(pred_fake, label_real)

    # pred average
    mean = pred_fake.mean()

    # backprop.
    loss.backward()

    return ResultPackage("Decoder_Loss",
                         dict(loss=loss.data[0], pred=mean.data[0]))


def train_gen(cfg, gen, ae, disc_c):
    gen.train()
    gen.zero_grad()

    noise = gen.make_noise(cfg)
    fake_code = gen(noise)
    err_g = disc_c(fake_code)

    # loss / backprop
    one = to_gpu(cfg.cuda, torch.FloatTensor([1]))
    err_g.backward(one)

    result = ResultPackage("Generator_Loss", dict(loss=err_g.data[0]))

    return result, fake_code


def train_disc_c(cfg, disc, ae, real_hidden, fake_hidden):
        # clamp parameters to a cube
    for p in disc.parameters():
        p.data.clamp_(-cfg.gan_clamp, cfg.gan_clamp) # [min,max] clamp
        # WGAN clamp (default:0.01)

    disc.train()
    disc.zero_grad()

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
                normed_grad = grad * ae.enc_grad_norm / gan_norm
            # grad : gradient from GAN
            # aeoder.grad_norm : norm(gradient from AE)
            # gan_norm : norm(gradient from GAN)
        else:
            normed_grad = grad

        # weight factor and sign flip
        normed_grad *= -math.fabs(cfg.gan_toenc)

        return normed_grad

    real_hidden.register_hook(grad_hook) # normed_grad
    # loss / backprop
    err_d_real = disc(real_hidden)
    one = to_gpu(cfg.cuda, torch.FloatTensor([1]))
    err_d_real.backward(one)

    # negative samples ----------------------------
    # loss / backprop
    err_d_fake = disc(fake_hidden.detach())
    err_d_fake.backward(one * -1)

    # `clip_grad_norm` to prvent exploding gradient problem in RNNs / LSTMs
    torch.nn.utils.clip_grad_norm(ae.parameters(), cfg.clip)

    err_d = -(err_d_real - err_d_fake)

    return ResultPackage("Code_GAN_Loss",
                         dict(D_Total=err_d.data[0],
                              D_Real=err_d_real.data[0],
                              D_Fake=err_d_fake.data[0]))


def train_disc_s(cfg, disc, in_real, in_fake, code_real, code_fake):
    disc.train()
    disc.zero_grad()

    if cfg.disc_s_in == 'embed':
        # in_real.size() : [batch_size, max_len]
        # in_fake.size() : [batch_size*2, max_len, vocab_size]
        # normal embedding lookup
        in_real = disc.embed(in_real)
        # soft embedding lookup (3D x 2D matrix multiplication)
        in_fake = disc.soft_embed(in_fake)
        # [batch_size, max_len, embed_size]

        # clamp parameters to a cube
    for p in disc.parameters():
        p.data.clamp_(-cfg.gan_clamp, cfg.gan_clamp) # [min,max] clamp
        # WGAN clamp (default:0.01)

    #in_real = in_real + in_real.eq(0).float() * (-1e-16)
    #in_fake = in_fake + in_fake.eq(0).float() * (-1e-16)

    rec_real, pred_real, attn_real = disc(in_real.detach())
    rec_fake, pred_fake, attn_fake = disc(in_fake.detach())

    # attention loss
    label_real = to_gpu(cfg.cuda, Variable(torch.ones(pred_real.size())))
    label_fake = to_gpu(cfg.cuda, Variable(torch.zeros(pred_fake.size())))
    l_gan_real = disc.criterion_bce(pred_real, label_real)
    l_gan_fake = disc.criterion_bce(pred_fake, label_fake)

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

    # backprop.
    l_real = l_gan_real + l_rec_real
    l_fake = l_gan_fake + l_rec_fake
    l_real.backward()
    l_fake.backward()

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
