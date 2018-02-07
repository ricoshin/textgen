import math
import collections

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

from train.train_helper import ResultPackage, append_pads
from utils.utils import to_gpu

dict = collections.OrderedDict


def train_ae(cfg, net, batch, ans_batch):
    # train encoder for answer
    net.ans_enc.train()
    # train ae
    net.enc.train()
    net.enc.zero_grad()
    net.dec.train()
    net.dec.zero_grad()
    # output.size(): batch_size x max_len x ntokens (logits)

    # output = answer encoder(ans_batch.src, ans_batch.len, noise=True, save_grad_norm=True)
    ans_code = net.ans_enc(ans_batch.src, ans_batch.len, noise=True, save_grad_norm=True)
    #output = ae(batch.src, batch.len, noise=True)
    code = net.enc(batch.src, batch.len, noise=True, save_grad_norm=True)
    output = net.dec(torch.cat((code, ans_code), 1), batch.src, batch.len) # torch.cat dim=1

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
    torch.nn.utils.clip_grad_norm(net.ans_enc.parameters(), cfg.clip)
    torch.nn.utils.clip_grad_norm(net.enc.parameters(), cfg.clip)
    torch.nn.utils.clip_grad_norm(net.dec.parameters(), cfg.clip)
    return ResultPackage("Autoencoder",
                         dict(Loss=loss.data, Accuracy=accuracy.data[0]))


def eval_ae_tf(net, batch, ans_code):
    net.enc.eval()
    net.dec.eval()

    # output.size(): batch_size x max_len x ntokens (logits)
    #output = ae(batch.src, batch.len, noise=True)
    code = net.enc(batch.src, batch.len, noise=True)
    output = net.dec(torch.cat((code, ans_code), 1), batch.src, batch.len)

    max_value, max_indices = torch.max(output, 2)
    target = batch.tar.view(output.size(0), -1)
    outputs = max_indices.data.cpu().numpy()
    targets = target.data.cpu().numpy()

    return targets, outputs

def eval_ae_fr(cfg, net, batch, ans_code):
    # forward / NOTE : ae_mode off?
    # "real" real
    code = net.enc(batch.src, batch.len, noise=False, train=False)
    max_ids, outputs = net.dec(torch.cat((code, ans_code), 1), teacher=False, train=False)
    # output.size(): batch_size x max_len x ntokens (logits)
    target = batch.tar.view(outputs.size(0), -1)
    targets = target.data.cpu().numpy()

def eval_ae_fr(net, batch, ans_code):
    net.enc.eval()
    net.dec.eval()
    # output.size(): batch_size x max_len x ntokens (logits)
    #code = ae.encode_only(cfg, batch, train=False)
    #max_ids, outs = ae.decode_only(cfg, code, vocab, train=False)

    code = net.enc(batch.src, batch.len, noise=True)
    max_ids, outs = net.dec.generate(torch.cat((code, ans_code), 1))

    targets = batch.tar.view(outs.size(0), -1)
    targets = targets.data.cpu().numpy()

    return targets, max_ids


def eval_gen_dec(cfg, net, fixed_noise, ans_code):
    net.gen.eval()
    net.dec.eval()
    code_fake = net.gen(fixed_noise)
    ids_fake, _ = net.dec.generate(torch.cat((code_fake, ans_code), 1))
    return ids_fake


def train_dec(cfg, net, fake_code, ans_code, vocab):
    net.dec.train()
    net.dec.zero_grad()

    fake_ids, fake_outs = net.dec.generate(torch.cat((fake_code, ans_code), 1))
    # fake_outs.size() : [batch_size*2, max_len, vocab_size]

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

        normed_grad *= math.fabs(cfg.gan_to_ae)
        return normed_grad

    net.dec.logits.register_hook(grad_hook)

    # loss
    pred_fake, attn_fake = net.disc_s(fake_outs)
    label_real = to_gpu(cfg.cuda, Variable(torch.ones(pred_fake.size())))
    loss = net.disc_s.criterion_bce(pred_fake, label_real)

    # pred average
    mean = pred_fake.mean()

    # backward
    loss.backward()

    return ResultPackage("Decoder_Loss",
                         dict(loss=loss.data[0], pred=mean.data[0]))


def generate_codes(cfg, net, batch):
    net.enc.train() # NOTE train encoder!
    net.enc.zero_grad()
    net.gen.eval()

    code_real = net.enc(batch.src, batch.len, noise=False)
    code_fake = net.gen(None)

    return code_real, code_fake

def train_disc_ans(cfg, net, code):


def train_disc_c(cfg, net, code_real, code_fake):
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
                normed_grad = grad
            else:
                normed_grad = grad * net.enc.grad_norm / gan_norm
            # grad : gradient from GAN
            # aeoder.grad_norm : norm(gradient from AE)
            # gan_norm : norm(gradient from GAN)
        else:
            normed_grad = grad

        # weight factor and sign flip
        normed_grad *= -math.fabs(cfg.gan_to_ae)

        return normed_grad

    code_real.register_hook(grad_hook) # normed_grad
    # loss / backprop
    err_d_real = net.disc_c(code_real)
    one = to_gpu(cfg.cuda, torch.FloatTensor([1]))
    err_d_real.backward(one)

    # negative samples ----------------------------
    # loss / backprop
    err_d_fake = net.disc_c(code_fake.detach())
    err_d_fake.backward(one * -1)

    # `clip_grad_norm` to prvent exploding gradient problem in RNNs / LSTMs
    torch.nn.utils.clip_grad_norm(net.enc.parameters(), cfg.clip)

    err_d = -(err_d_real - err_d_fake)

    return ResultPackage("Code_GAN_Loss",
               dict(D_Total=err_d.data[0],
                    D_Real=err_d_real.data[0],
                    D_Fake=err_d_fake.data[0]))
