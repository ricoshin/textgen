import math

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

from train.train_helper import ResultPackage
from utils.utils import to_gpu


def train_ae(cfg, net, batch):
    net.enc.train()
    net.enc.zero_grad()
    net.dec.train()
    net.dec.zero_grad()

    # output.size(): batch_size x max_len x ntokens (logits)

    #output = ae(batch.src, batch.len, noise=True)
    code = net.enc(batch.src, batch.len, noise=True, save_grad_norm=True)
    output = net.dec(code, batch.src, batch.len)

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

    loss = net.dec.criterion_ce(masked_output, masked_target)

    loss.backward()
    # `clip_grad_norm` to prevent exploding gradient in RNNs / LSTMs
    torch.nn.utils.clip_grad_norm(net.enc.parameters(), cfg.clip)
    torch.nn.utils.clip_grad_norm(net.dec.parameters(), cfg.clip)
    return ResultPackage("Autoencoder",
                         dict(Loss=loss.data, Accuracy=accuracy.data[0]))


def eval_ae_tf(net, batch):
    net.enc.eval()
    net.dec.eval()
    # output.size(): batch_size x max_len x ntokens (logits)
    #output = ae(batch.src, batch.len, noise=True)
    code = net.enc(batch.src, batch.len, noise=True)
    output = net.dec(code, batch.src, batch.len)

    max_value, max_indices = torch.max(output, 2)
    target = batch.tar.view(output.size(0), -1)
    outputs = max_indices.data.cpu().numpy()
    targets = target.data.cpu().numpy()

    return targets, outputs


def eval_ae_fr(net, batch):
    net.enc.eval()
    net.dec.eval()
    # output.size(): batch_size x max_len x ntokens (logits)
    #code = ae.encode_only(cfg, batch, train=False)
    #max_ids, outs = ae.decode_only(cfg, code, vocab, train=False)

    code = net.enc(batch.src, batch.len, noise=True)
    max_ids, outs = net.dec.generate(code)

    targets = batch.tar.view(outs.size(0), -1)
    targets = targets.data.cpu().numpy()

    return targets, max_ids


def eval_gen_dec(cfg, net, fixed_noise):
    net.gen.eval()
    net.dec.eval()
    fake_hidden = net.gen.generate(cfg, fixed_noise, False)
    ids_fake, _ = net.dec.generate(fake_hidden)
    return ids_fake


def train_dec(cfg, net, fake_code, vocab):
    net.dec.train()
    net.dec.zero_grad()

    fake_ids, fake_outs = net.dec(fake_code)
    # fake_outs.size() : [batch_size*2, max_len, vocab_size]

    if cfg.disc_s_in == 'embed':
        # fake_in.size() : [batch_size*2, max_len, vocab_size]
        # soft embedding lookup (3D x 2D matrix multiplication)
        fake_outs = net.disc_s.soft_embed(fake_outs)
        # [batch_size, max_len, embed_size]

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

        normed_grad *= math.fabs(cfg.gan_toenc)
        return normed_grad

    fake_outs.register_hook(grad_hook)

    # loss
    _, pred_fake, attn_fake = net.disc_s(fake_outs)
    label_real = to_gpu(cfg.cuda, Variable(torch.ones(pred_fake.size())))
    loss = net.disc_s.criterion_bce(pred_fake, label_real)

    # pred average
    mean = pred_fake.mean()

    # backprop.
    loss.backward()

    return ResultPackage("Decoder_Loss",
                         dict(loss=loss.data[0], pred=mean.data[0]))


def train_gen(cfg, net):
    net.gen.train()
    net.gen.zero_grad()

    noise = net.gen.make_noise(cfg)
    fake_code = net.gen(noise)
    err_g = net.disc_c(fake_code)

    # loss / backprop
    one = to_gpu(cfg.cuda, torch.FloatTensor([1]))
    err_g.backward(one)

    result = ResultPackage("Generator_Loss", dict(loss=err_g.data[0]))

    return result, fake_code


def generate_codes(cfg, net, batch):
    net.enc.train() # NOTE train encoder!
    net.enc.zero_grad()
    net.gen.eval()

    code_real = net.enc(batch.src, batch.len, noise=False)
    code_fake = net.gen.generate(cfg, None, False)

    return code_real, code_fake


def train_disc_c(cfg, net, real_hidden, fake_hidden):
    # clamp parameters to a cube
    for p in net.disc_c.parameters():
        p.data.clamp_(-cfg.gan_clamp, cfg.gan_clamp) # [min,max] clamp
        # WGAN clamp (default:0.01)

    net.disc_c.train()
    net.disc_c.zero_grad()

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

        # weight factor and sign flip
        normed_grad *= -math.fabs(cfg.gan_toenc)

        return normed_grad

    real_hidden.register_hook(grad_hook) # normed_grad
    # loss / backprop
    err_d_real = net.disc_c(real_hidden)
    one = to_gpu(cfg.cuda, torch.FloatTensor([1]))
    err_d_real.backward(one)

    # negative samples ----------------------------
    # loss / backprop
    err_d_fake = net.disc_c(fake_hidden.detach())
    err_d_fake.backward(one * -1)

    # `clip_grad_norm` to prvent exploding gradient problem in RNNs / LSTMs
    torch.nn.utils.clip_grad_norm(net.enc.parameters(), cfg.clip)

    err_d = -(err_d_real - err_d_fake)

    return ResultPackage("Code_GAN_Loss",
               dict(D_Total=err_d.data[0],
                    D_Real=err_d_real.data[0],
                    D_Fake=err_d_fake.data[0]))


def train_disc_s(cfg, code_real, code_fake):
    net.dec.eval()
    net.disc_s.train()
    net.disc_s.zero_grad()

    ids_real, outs_real = net.dec.generate(code_real)
    ids_fake, outs_fake = net.dec.generate(code_fake)

    if cfg.disc_s_in == 'embed':
        ids_fake = [ids_real, ids_fake]
        # "real" fake (embeddings)
        outs_fake = torch.cat([outs_real, outs_fake], dim=0)
        code_fake = torch.cat([code_real, code_fake], dim=0)
        # "real" real ids
        ids_real = batch.tar.view(cfg.batch_size, -1)
        ids_real = append_pads(cfg, ids_real, net.vocab)
        #outs_real = batch.tar.view(cfg.batch_size, -1)

        # normal embedding lookup
        # ids_real.size() : [batch_size, max_len]
        in_real = net.disc_s.embed(ids_real)
        # soft embedding lookup (3D x 2D matrix multiplication)
        # in_fake.size() : [batch_size*2, max_len, vocab_size]
        in_fake = net.disc_s.soft_embed(outs_fake)
        # [batch_size, max_len, embed_size]

    # clamp parameters to a cube
    for p in net.disc_s.parameters():
        p.data.clamp_(-cfg.gan_clamp, cfg.gan_clamp) # [min,max] clamp
        # WGAN clamp (default:0.01)

    #in_real = in_real + in_real.eq(0).float() * (-1e-16)
    #in_fake = in_fake + in_fake.eq(0).float() * (-1e-16)

    rec_real, pred_real, attn_real = net.disc_s(in_real.detach())
    rec_fake, pred_fake, attn_fake = net.disc_s(in_fake.detach())

    # attention loss
    label_real = to_gpu(cfg.cuda, Variable(torch.ones(pred_real.size())))
    label_fake = to_gpu(cfg.cuda, Variable(torch.zeros(pred_fake.size())))
    l_gan_real = net.disc_s.criterion_bce(pred_real, label_real)
    l_gan_fake = net.disc_s.criterion_bce(pred_fake, label_fake)

    # total loss
    l_gan_total = l_gan_real + l_gan_fake
    l_total = l_gan_total + l_rec_total

    # pred mean
    real_mean = pred_real.mean()
    fake_mean = pred_fake.mean()

    # backprop.
    l_gan_real.backward()
    l_gan_fake.backward()

    loss_gan = ResultPackage("Sample_GAN_loss",
                             dict(D_Total=l_gan_total,
                                  D_Real=l_gan_real,
                                  D_Fake=l_gan_fake))
    pred_gan = ResultPackage("Sample_GAN_pred",
                             dict(D_real=real_mean,
                                  D_Fake=fake_mean))

    ids = [ids_real, ids_fake]
    attns = [attn_real, attn_fake]

    return loss_gan, pred_gan, ids, attns
