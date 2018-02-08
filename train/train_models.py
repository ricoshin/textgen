import math
import collections

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

from train.train_helper import ResultPackage, append_pads
from utils.utils import to_gpu

dict = collections.OrderedDict


def train_ae(cfg, net, batch):
    # train encoder for answer
    net.ans_enc.train()
    # train ae
    net.enc.train()
    net.enc.zero_grad()
    net.dec.train()
    net.dec.zero_grad()
    # output.size(): batch_size x max_len x ntokens (logits)

    # output = answer encoder(ans_batch.src, ans_batch.len, noise=True, save_grad_norm=True)
    ans_code = net.ans_enc(batch.a, batch.a_len, noise=True, save_grad_norm=True)
    #output = ae(batch.src, batch.len, noise=True)
    code = net.enc(batch.q, batch.q_len, noise=True, save_grad_norm=True)
    output = net.dec(torch.cat((code, ans_code), 1), batch.q, batch.q_len) # torch.cat dim=1

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
        mask_output_target(output, batch.q_tar, cfg.vocab_size)

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
    code = net.enc(batch.q, batch.q_len, noise=True)
    output = net.dec(torch.cat((code, ans_code), 1), batch.q, batch.q_len)

    max_value, max_indices = torch.max(output, 2)
    target = batch.q_tar.view(output.size(0), -1)
    outputs = max_indices.data.cpu().numpy()
    targets = target.data.cpu().numpy()

    return targets, outputs

def eval_ae_fr(cfg, net, batch, ans_code):
    # forward / NOTE : ae_mode off?
    # "real" real
    code = net.enc(batch.q, batch.q_len, noise=False, train=False)
    max_ids, outputs = net.dec(torch.cat((code, ans_code), 1), teacher=False, train=False)
    # output.size(): batch_size x max_len x ntokens (logits)
    target = batch.q_tar.view(outputs.size(0), -1)
    targets = target.data.cpu().numpy()

def eval_ae_fr(net, batch, ans_code):
    net.enc.eval()
    net.dec.eval()
    # output.size(): batch_size x max_len x ntokens (logits)
    #code = ae.encode_only(cfg, batch, train=False)
    #max_ids, outs = ae.decode_only(cfg, code, vocab, train=False)

    code = net.enc(batch.q, batch.q_len, noise=True)
    max_ids, outs = net.dec.generate(torch.cat((code, ans_code), 1))

    targets = batch.q_tar.view(outs.size(0), -1)
    targets = targets.data.cpu().numpy()

    return targets, max_ids

def train_disc_ans(cfg, net, batch):
    # make answer encoding
    net.ans_enc.train() # train answer encoder
    net.ans_enc.zero_grad()
    ans_code = net.ans_enc(batch.a, batch.a_len, noise=True, save_grad_norm=True)

    # train answer discriminator
    net.disc_ans.train()
    net.disc_ans.zero_grad()
    logit = net.disc_ans(batch.q, batch.q_len)

    # calculate loss and backpropagate
    # logit : (N=question sent len, C=answer embed size)
    # ans_code : C=answer embed size
    ans_code = Variable(ans_code.data, requires_grad = False)
    KLD_loss = torch.nn.modules.loss.KLDivLoss()
    loss = KLD_loss(logit, ans_code) # target : label, text : feature
    loss.backward()
    torch.nn.utils.clip_grad_norm(net.disc_ans.parameters(), cfg.clip)

    # calculate accuracy
    """
    need to add functionality
    """

    return logit, loss

def eval_disc_ans(net, batch, ans_code):
    net.disc_ans.eval()
    logit = net.disc_ans(batch.q, batch.q_len)

    # calculate kl divergence loss
    ans_code = Variable(ans_code.data, requires_grad = False)
    KLD_loss = torch.nn.modules.loss.KLDivLoss()
    loss = KLD_loss(logit, ans_code) # target : label, text : feature
    # print discriminator output
    code = net.enc(batch.q, batch.q_len, noise=True)
    output = net.dec(torch.cat((code, logit), 1), batch.q, batch.q_len)
    max_value, max_indices = torch.max(output, 2)
    target = batch.q_tar.view(output.size(0), -1)
    outputs = max_indices.data.cpu().numpy()
    targets = target.data.cpu().numpy()
    return logit, loss, targets, outputs
