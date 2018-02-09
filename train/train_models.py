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


def train_ae(cfg, net, batch, mode):
    net.embed.train()
    net.embed.zero_grad()
    net.enc.train()
    net.enc.zero_grad()
    net.dec.train()
    net.dec.zero_grad()
    # output.size(): batch_size x max_len x ntokens (logits)

    in_embed = net.embed(batch.src)
    code = net.enc(in_embed, noise=True, save_grad_norm=True)

    if mode == 'tf':
        out_word, _, out_tag = net.dec(code, batch.src, batch.len, mode='tf')
    elif mode == 'fr':
        out_word, _, out_tag = net.dec(code, batch.len, mode='fr',
                                       save_grad_norm=True)
    else:
        raise Exception("Unknown decoder training mode!")


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
    cosim = compute_cosine_sim(out_word, net.embed)
    out_word_p = F.log_softmax(cosim * cfg.embed_temp, 2)
    msk_out, msk_tar = mask_output_target(out_word_p, batch.tar, cfg.vocab_size)
    loss_word = net.dec.criterion_nll(msk_out, msk_tar)
    _, max_ids = torch.max(msk_out, 1)
    acc_word = torch.mean(max_ids.eq(msk_tar).float())

    # compute tag prediction loss and accuracy
    msk_out, msk_tar = mask_output_target(out_tag, batch.tar_tag, cfg.tag_size)
    loss_tag = net.dec.criterion_ce(msk_out, msk_tar)
    _, max_ids = torch.max(msk_out, 1)
    acc_tag = torch.mean(max_ids.eq(msk_tar).float())

    loss_word.backward(retain_graph=True)
    loss_tag.backward()
    loss_total = loss_word + loss_tag
    #loss_total.backward()

    # `clip_grad_norm` to prevent exploding gradient in RNNs / LSTMs
    torch.nn.utils.clip_grad_norm(net.enc.parameters(), cfg.clip)
    torch.nn.utils.clip_grad_norm(net.dec.parameters(), cfg.clip)

    if mode == 'tf':
        name = "Autoencoder"
    elif mode == 'fr':
        name = "Autoencoder_fr"
    return ResultPackage(name, dict(Loss_word=loss_word.data,
                                    Loss_tag=loss_tag.data,
                                    Acc_word=acc_word.data[0],
                                    Acc_tag=acc_tag.data[0]))


def train_exposure(cfg, net, batch):
    net.embed.eval()
    net.enc.eval()
    net.dec.train()
    net.dec.zero_grad()

    # encode
    in_embed = net.embed(batch.src)
    code = net.enc(in_embed, noise=False, save_grad_norm=False)

    # decode
    code_detached = Variable(code.data, requires_grad=False)
    out_word_tf, _, out_tag = net.dec(code_detached, batch, mode='tf')
    out_word_fr, _, out_tag = net.dec(code_detached, batch.len, mode='fr')

    # encode again
    code_tf = net.enc(out_word_tf, noise=False, save_grad_norm=False)
    code_fr = net.enc(out_word_fr, noise=False, save_grad_norm=False)

    code_tar = Variable(code_tf.data, requires_grad=False)
    # [bsz, hidden_size]
    bsz = code_tar.size(0)
    # trick for batch-wise dot product
    similarity = torch.bmm(code_tar.view(bsz, 1, -1), code_fr.view(bsz, -1, 1))
    loss = torch.mean(similarity)
    #loss = net.enc.criterion_mse(code_fr, code_tar)
    loss.backward()


def train_enc(cfg, net, batch):
    net.embed.eval()
    net.enc.train()
    net.enc.zero_grad()
    net.dec.train()
    net.enc.zero_grad()

    # code reconstruction loss
    in_embed = net.embed(batch.src)
    code = net.enc(in_embed, noise=False, save_grad_norm=False)
    out_word, _, out_tag = net.dec(code, batch.src, batch.len, mode='tf')
    code_recon = net.enc(out_word, noise=False, save_grad_norm=False)

    code_detached = Variable(code.data, requires_grad=False)
    loss = net.enc.criterion_mse(code_recon, code_detached)
    loss.backward()


def eval_ae_tf(net, batch):
    net.enc.eval()
    net.dec.eval()
    # output.size(): batch_size x max_len x ntokens (logits)
    #output = ae(batch.src, batch.len, noise=True)
    in_embed = net.embed(batch.src)
    code = net.enc(in_embed, noise=True)
    _, out_word_p, _ = net.dec(code, batch.src, batch.len, mode='tf') #NOTE

    max_value, max_indices = torch.max(out_word_p, 2)
    target = batch.tar.view(out_word_p.size(0), -1)
    outputs = max_indices.data.cpu().numpy()
    targets = target.data.cpu().numpy()

    return targets, outputs


def eval_ae_fr(net, batch):
    net.enc.eval()
    net.dec.eval()
    # output.size(): batch_size x max_len x ntokens (logits)
    #code = ae.encode_only(cfg, batch, train=False)
    #max_ids, outs = ae.decode_only(cfg, code, vocab, train=False)
    in_embed = net.embed(batch.src)
    code = net.enc(in_embed, noise=True)
    _, out_word_p, _ = net.dec(code, batch.len, mode='fr')

    max_value, max_indices = torch.max(out_word_p, 2)
    target = batch.tar.view(out_word_p.size(0), -1)
    outputs = max_indices.data.cpu().numpy()
    targets = target.data.cpu().numpy()

    return targets, outputs


def eval_gen_dec(cfg, net, fixed_noise):
    net.gen.eval()
    net.dec.eval()
    code_fake = net.gen(fixed_noise)
    _, out_word_p, _ = net.dec(code_fake,  mode='gen')
    _, max_indices = torch.max(out_word_p, 2)
    max_indices = max_indices.data.cpu().numpy()

    return max_indices


# def train_dec(cfg, net, fake_code, vocab):
#     net.dec.train()
#     net.dec.zero_grad()
#
#     fake_ids, _, fake_outs = net.dec.generate(fake_code)
#     # fake_outs.size() : [batch_size*2, max_len, vocab_size]
#
#     def grad_hook(grad):
#         if cfg.ae_grad_norm:
#             gan_norm = torch.norm(grad, 2, 1).detach().data.mean()
#             if gan_norm == .0:
#                 log.warning("zero sample_gan norm!")
#                 import pdb; pdb.set_trace()
#                 normed_grad = grad
#             else:
#                 normed_grad = grad * net.dec.grad_norm / gan_norm
#         else:
#             normed_grad = grad
#
#         normed_grad *= math.fabs(cfg.gan_to_ae)
#         return normed_grad
#
#     if net.dec.gard_norm is None:
#         log.warning("Decoder gradient has never been saved!")
#     else:
#         net.dec.logits.register_hook(grad_hook)
#
#     # loss
#     pred_fake, attn_fake = net.disc_s(fake_outs)
#     label_real = to_gpu(cfg.cuda, Variable(torch.ones(pred_fake.size())))
#     loss = net.disc_s.criterion_bce(pred_fake, label_real)
#
#     # pred average
#     mean = pred_fake.mean()
#
#     # backward
#     loss.backward()
#
#     return ResultPackage("Decoder_Loss",
#                          dict(loss=loss.data[0], pred=mean.data[0]))


def train_gen(cfg, net):
    net.gen.train()
    net.gen.zero_grad()

    fake_code = net.gen(None)
    loss, pred = net.disc_c(fake_code)

    # loss / backprop
    one = to_gpu(cfg.cuda, torch.FloatTensor([1]))
    loss.backward(one, retain_graph=True)
    pred_mean = pred.mean()

    result = ResultPackage("Generator_Loss",
                           dict(loss=loss.data[0],
                                pred=pred_mean.data[0]))

    return result, fake_code


def train_dec(cfg, net, code_fake):
    net.enc.eval()
    net.disc_s.eval()
    net.dec.train()
    net.dec.zero_grad()

    #code_fake = Variable(code_fake.data, requires_grad=False)
    embed_fake, _, _ = net.dec(code_fake, mode='gen')
    code_fake_r = net.enc(embed_fake, noise=False)

    def grad_hook(grad):
        if cfg.ae_grad_norm:
            gan_norm = torch.norm(grad, 2, 1).detach().data.mean()
            if gan_norm == .0:
                log.warning("zero sample_gan norm!")
                import pdb; pdb.set_trace()
                normed_grad = grad
            else:
                normed_grad = grad * net.dec.grad_norm / gan_norm
        else:
            normed_grad = grad

        #normed_grad *= math.fabs(cfg.gan_to_ae)
        return normed_grad

    if net.dec.grad_norm is None:
        log.warning("Decoder gradient has never been saved!")
    else:
        embed_fake.register_hook(grad_hook)

    loss, pred = net.disc_s(code_fake_r)

    # loss / backprop
    one = to_gpu(cfg.cuda, torch.FloatTensor([1]))
    loss.backward(one)

    return ResultPackage("Decoder_Loss",
                         dict(loss=loss.data[0]))


def generate_codes(cfg, net, batch):
    net.enc.train() # NOTE train encoder!
    net.enc.zero_grad()
    net.gen.eval()

    in_embed = net.embed(batch.src)
    code_real = net.enc(in_embed, noise=False)
    code_fake = net.gen(None)

    return code_real, code_fake


def train_disc_c(cfg, net, code_real, code_fake):
    # clamp parameters to a cube
    for name, params in net.disc_c.named_parameters():
        if 'pred_linear' not in name:
            params.data.clamp_(-cfg.gan_clamp, cfg.gan_clamp) # [min,max] clamp
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
                import pdb; pdb.set_trace()
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

    # feed real & fake codes
    loss_real, pred_real = net.disc_c(code_real)
    loss_fake, pred_fake = net.disc_c(code_fake.detach())

    # WGAN backward
    one = to_gpu(cfg.cuda, torch.FloatTensor([1]))
    loss_real.backward(one)
    loss_fake.backward(one * -1)
    loss_total = loss_real - loss_fake

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

    # `clip_grad_norm` to prvent exploding gradient problem in RNNs / LSTMs
    torch.nn.utils.clip_grad_norm(net.enc.parameters(), cfg.clip)

    return ResultPackage("Code_GAN_Loss",
                         dict(D_Loss_Total=loss_total.data[0],
                              D_Loss_Real=loss_real.data[0],
                              D_Loss_Fake=loss_fake.data[0]))
                              # D_Pred_Real=pred_real_mean.data[0],
                              # D_Pred_Fake=pred_fake_mean.data[0]))


def train_disc_s(cfg, net, batch, code_real, code_fake):
    # clamp parameters to a cube
    for name, params in net.disc_s.named_parameters():
        if 'pred_linear' not in name:
            params.data.clamp_(-cfg.gan_clamp, cfg.gan_clamp) # [min,max] clamp
        # WGAN clamp (default:0.01)

    net.enc.eval()
    net.dec.eval()
    net.disc_s.train()
    net.disc_s.zero_grad()

    # real pipeline
    embed_real = net.embed(batch.src)
    code_real = net.enc(embed_real, noise=False, save_grad_norm=False)
    code_real = Variable(code_real.data, requires_grad=False) # cut the graph

    # fake pipeline
    embed_fake, _, _ = net.dec(code_fake, mode='gen')
    code_fake_r = net.enc(embed_fake, noise=False, save_grad_norm=False)
    code_fake_r = Variable(code_fake_r.data, requires_grad=False)

    # sample discriminator
    # feed real & fake codes
    loss_real, pred_real = net.disc_s(code_real)
    loss_fake, pred_fake = net.disc_s(code_fake_r)

    # WGAN backward
    one = to_gpu(cfg.cuda, torch.FloatTensor([1]))
    loss_real.backward(one)
    loss_fake.backward(one * -1)
    loss_total = loss_real - loss_fake

    torch.nn.utils.clip_grad_norm(net.enc.parameters(), cfg.clip)

    return ResultPackage("Sample_GAN_Loss",
                         dict(D_Loss_Total=loss_total.data[0],
                              D_Loss_Real=loss_real.data[0],
                              D_Loss_Fake=loss_fake.data[0]))


# def train_disc_s(cfg, net, batch, code_real, code_fake):
#     pass
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


def train_disc_d(cfg, net, batch, code_real, code_fake):
    net.dec.eval()
    net.disc_d.train()
    net.disc_d.zero_grad()

    ids_real, _, outs_real = net.dec(code_real, mode='fr') #NOTE from here!
    ids_fake, _, outs_fake = net.dec(code_fake, mode='fr')

    # "real" fake (embeddings)
    outs_fake = torch.cat([outs_real, outs_fake], dim=0)
    code_fake = torch.cat([code_real, code_fake], dim=0)

    # clamp parameters to a cube
    for p in net.disc_s.parameters():
        p.data.clamp_(-cfg.gan_clamp, cfg.gan_clamp) # [min,max] clamp
        # WGAN clamp (default:0.01)

    pred_real, attn_real = net.disc_s(batch.src.detach())
    pred_fake, attn_fake = net.disc_s(outs_fake.detach())

    # GAN loss
    label_real = to_gpu(cfg.cuda, Variable(torch.ones(pred_real.size())))
    label_fake = to_gpu(cfg.cuda, Variable(torch.zeros(pred_fake.size())))
    loss_real = net.disc_s.criterion_bce(pred_real, label_real)
    loss_fake = net.disc_s.criterion_bce(pred_fake, label_fake)
    loss_total = loss_real + loss_fake

    # pred mean
    real_mean = pred_real.mean()
    fake_mean = pred_fake.mean()

    # backprop.
    loss_real.backward()
    loss_fake.backward()

    # results
    loss_gan = ResultPackage("Sample_GAN_loss",
                             dict(D_Total=loss_total,
                                  D_Real=loss_real,
                                  D_Fake=loss_fake))
    pred_gan = ResultPackage("Sample_GAN_pred",
                             dict(D_real=real_mean,
                                  D_Fake=fake_mean))

    ids = [batch.src.data.cpu().numpy(), ids_fake]
    attns = [attn_real, attn_fake]

    return loss_gan, pred_gan, ids, attns
