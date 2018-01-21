import logging
import numpy as np
import os
import time

from tensorboardX import SummaryWriter
import torch
import torch.nn as nn
from torch.autograd import Variable

from train_helper import TrainingSupervisor
from utils import set_random_seed, to_gpu

from autoencoder import Autoencoder
from code_disc import CodeDiscriminator
from evaluate import evaluate_sents
from generator import Generator
from sample_disc import SampleDiscriminator

log = logging.getLogger('main')


def print_line(char='-', row=1, length=130):
    for i in range(row):
        log.info(char * length)

def ids_to_sent(vocab, ids, length=None, no_pad=True):
    if length is None:
        length = 999
    if no_pad:
        return " ".join([vocab.idx2word[idx] for i, idx in enumerate(ids)
                         if idx != vocab.PAD_ID and i < length])
    else:
        return " ".join([vocab.idx2word[idx] for i, idx in enumerate(ids)
                         if i < length])

def pad_after_eos(vocab, ids, last_eos=True):
    truncated_ids = []
    eos = False
    for idx in ids:
        if (not eos) and (idx == vocab.EOS_ID):
            if last_eos:
                truncated_ids.append(idx)
            eos = True
            continue
        if not eos:
            truncated_ids.append(idx)
        else:
            truncated_ids.append(vocab.PAD_ID)
    return truncated_ids

def ids_to_sent_for_eval(vocab, ids):
    sents = []
    for idx in ids:
        truncated_ids = pad_after_eos(vocab, idx, last_eos=False)
        sents.append(ids_to_sent(vocab, truncated_ids, no_pad=True))
    return sents

def print_ae_sents(vocab, target_ids, output_ids, lengths, nline=5):
    coupled = list(zip(target_ids, output_ids, lengths))
    # shuffle : to prevent always printing the longest ones first
    np.random.shuffle(coupled)
    print_line()
    for i, (tar_ids, out_ids, length) in enumerate(coupled):
        if i > nline - 1: break
        log.info("[X] " + ids_to_sent(vocab, tar_ids, length=length))
        log.info("[Y] " + ids_to_sent(vocab, out_ids, length=length))
        print_line()

def print_info(sv):
    print_line()
    log.info("| Name : %s | Epoch : %d/%d | Batches : %d/%d |"
             % (sv.cfg.name, sv.epoch_step, sv.epoch_total,
                sv.batch_step, sv.batch_total))

def print_nums(sv, title, num_dict):
    print_str = "| %s |" % title
    for key, value in num_dict.items():
        print_str += " %s : %.8f |" % (key, value)
    log.info(print_str)

def print_gen_sents(vocab, output_ids, nline=999):
    print_line()
    for i, ids in enumerate(output_ids):
        if i > nline - 1: break
        #ids = pad_after_eos(vocab, ids)
        log.info(ids_to_sent(vocab, ids))
        print_line()
    print_line(' ')

def align_word_attn(words, attns_w, attns_l, min_width=4):
    # attn_list[i] : [attn1[i], attn2[i], attn3[i]]
    word_formats = ' '.join(['{:^%ds}' % max(min_width, len(word))
                            for word in words])
    word_str = word_formats.format(*words)
    attn_str_list = []
    # group word & layer attention by layers
    for (attn_w, attn_l) in zip(attns_w, attns_l):
        attn_formats = ' '.join(['{:^%d}' % max(min_width, len(word))
                                 for word in words])
        attn_w = [int(a*100) for a in attn_w]
        attn_str = attn_formats.format(*attn_w)
        attn_str = attn_str.replace('-100', '    ') # remove empty slots
        attn_str += "  [ %5.4f ]" % attn_l # append layer-wise attend
        attn_str_list.append(attn_str)
    return word_str, attn_str_list

def mark_empty_attn_w(attns, max_len):
    filter_n_stride = [(3,1), (3,2), (3,2), (4,1)]
    #attns = list(zip(*attns)) # layer first
    assert len(filter_n_stride) == len(attns)
    filters, strides = zip(*filter_n_stride)
    stride_ = 1
    actual_strides = []
    for stride in strides:
        stride_ *= stride
        actual_strides.append(stride_) # 1, 2, 4
    left_empty = 0
    actual_stride = 1
    new_attns = []
    for i, attn in enumerate(attns):
        # layerwise
        if i == 0:
            prev_stride = 1
        else:
            prev_stride = strides[i-1]
        left_empty += (filters[i] // 2) * prev_stride
        attn = np.array(attn)
        new_attn = np.ones([attn.shape[0], left_empty]) * (-1)
        empty_attn = np.ones([attn.shape[0], 1]) * (-1) # for column inserting
        attn_cnt = 0
        actual_strides *= strides[i]
        for j in range(max_len - left_empty):
            if j % actual_strides[i]  == 0 and attn_cnt < attn.shape[1]:
                new_attn = np.append(new_attn, attn[:, [attn_cnt]], axis=1)
                attn_cnt += 1
            else:
                new_attn = np.append(new_attn, empty_attn, axis=1)
        new_attns.append(new_attn)
        # [array(attn_1), array(att_2), array(attn_3)]
    return new_attns

def batch_first_attns(attns):
    # [[array(attn_1)[0], array(att_2)[0], array(attn_3)[0]],
    #  [array(attn_1)[1], array(att_2)[1], array(attn_3)[1]],
    #                        ......... (batch_size)        ]
    attns_w, attns_s = attns
    return (list(zip(*attns_w)), list(zip(*attns_s)))

def halve_attns(attns):
    attns_w, attns_l = attns
    attns_w, attns_l = list(zip(*attns_w)), list(zip(*attns_l))
    whole_batch = list(zip(attns_w, attns_l))
    first_half = whole_batch[len(whole_batch)//2:]
    first_w, first_l = list(zip(*first_half))
    first_half = list(zip(*first_w)), list(zip(*first_l))
    second_half = whole_batch[:len(whole_batch)//2]
    second_w, second_l = list(zip(*second_half))
    second_half = list(zip(*second_w)), list(zip(*second_l))
    return [first_half, second_half]

def print_attns(cfg, vocab, id_attn_pair):
    # len(attns_real) : batch_size
    # attns_real[0] : [array(attn_1[0]), array(attn_2[0], array(attn_3[0]))
    def print_aligned(bat_ids, bat_attns):
        sample_num = 2
        attns_w, attns_l = bat_attns
        #import ipdb; ipdb.set_trace()
        attns_w = mark_empty_attn_w(attns_w, cfg.max_len + 1)
        attns_w, attns_l = batch_first_attns([attns_w, attns_l])

        for i, sent_wise in enumerate(zip(bat_ids, attns_w, attns_l)):
            # sentenwise in a batch
            ids, attns_w, attns_l = sent_wise
            if i > sample_num - 1: break
            # ids = pad_after_eos(vocab, ids) # redundant for real_ids
            words = [vocab.idx2word[idx] for idx in ids]
            word_str, attn_str_list = align_word_attn(words, attns_w, attns_l)
            for attn_str in reversed(attn_str_list): # from topmost attn layer
                log.info(attn_str)
            log.info(word_str)
            print_line()

    for name, (ids, attns) in id_attn_pair.items():
        # id_attn_pair : dict(tuple(ids: attns))
        print_line()
        log.info("Attention on %s samples" % name)
        print_line()
        print_aligned(ids, attns)
    print_line(' ')

def load_test_data(cfg):
    test_sents = []
    with open(os.path.join(cfg.data_dir, 'test.txt')) as f:
        for line in f:
            test_sents.append(line.strip())
    return test_sents

def append_pads(cfg, tensor, vocab):
    pad_len = (cfg.max_len+1) - tensor.size(1)
    if pad_len > 0:
        pads = torch.ones([cfg.batch_size, pad_len]) * vocab.PAD_ID
        pads = Variable(pads, requires_grad=False).long().cuda()
        return torch.cat((tensor, pads), dim=1)
    else:
        return tensor

def mask_sequence_with_n_inf(self, seqs, seq_lens):
    max_seq_len = seqs.size(1)
    masks = seqs.data.new(*seqs.size()).zero_()
    for mask, seq_len in zip(masks, seq_lens):
        seq_len = seq_len.data[0]
        if seq_len < max_seq_len:
            mask[seq_len:] = float('-inf')
    masks = Variable(masks, requires_grad=False)
    return seqs + masks

def train(net):
    log.info("Training start!")
    cfg = net.cfg # for brevity
    set_random_seed(cfg)
    fixed_noise = Generator.make_noise_(cfg, cfg.eval_size) # for generator
    writer = SummaryWriter(cfg.log_dir)
    sv = TrainingSupervisor(net)
    test_sents = load_test_data(cfg)

    while not sv.global_stop():
        while not sv.epoch_stop():

            # train autoencoder ----------------------------
            for i in range(cfg.niters_ae): # default: 1 (constant)
                if sv.epoch_stop():
                    break  # end of epoch
                batch = net.data_ae.next()
                ae_loss, ae_acc = Autoencoder.train_(cfg, net.ae, batch)
                net.optim_ae.step()
                sv.inc_batch_step()

            # train gan ----------------------------------
            for k in range(sv.gan_niter): # epc0=1, epc2=2, epc4=3, epc6=4

                # train discriminator/critic (at a ratio of 5:1)
                for i in range(cfg.niters_gan_d): # default: 5
                    # feed a seen sample within this epoch; good for early training
                    # randomly select single batch among entire batches in the epoch
                    batch = net.data_gan.next()

                    # train CodeDiscriminator
                    code_real = Autoencoder.encode_(cfg, net.ae, batch)
                    code_fake = Generator.generate_(cfg, net.gen, None, False)
                    err_dc = CodeDiscriminator.train_(
                        cfg, net.disc_c, net.ae, code_real, code_fake)
                    err_dc_total, err_dc_real, err_dc_fake = err_dc

                    # train SampleDiscriminator
                    if cfg.with_attn and sv.epoch_step >= cfg.disc_s_hold:
                        ids_real, real_outs = \
                            Autoencoder.decode_(cfg, net.ae, code_real,
                                                net.vocab)
                        ids_fake, fake_outs = \
                            Autoencoder.decode_(cfg, net.ae, code_fake,
                                                net.vocab)
                        if cfg.disc_s_in == 'embed':
                            # "real" fake
                            fake_outs = torch.cat([real_outs, fake_outs], dim=0)
                            # "real" real
                            real_outs = batch.tar.view(cfg.batch_size, -1)
                            real_outs = append_pads(cfg, real_outs, net.vocab)
                            #real_outs = batch.tar.view(cfg.batch_size, -1)

                        loss_ds, acc_ds, attns = SampleDiscriminator.train_(
                            cfg, net.disc_s, real_outs, fake_outs)

                        net.optim_disc_s.step()

                    net.optim_ae.step()
                    net.optim_disc_c.step()

                # train generator(with disc_c) / decoder(with disc_s)
                for i in range(cfg.niters_gan_g): # default: 1
                    err_g, code_fake = Generator.train_(cfg, net.gen, net.ae,
                                                        net.disc_c)
                    if cfg.with_attn and sv.epoch_step >= cfg.disc_s_hold:
                        loss_dec, acc_dec = Autoencoder.decoder_train_(
                            cfg, net.ae, net.disc_s, code_fake, net.vocab)
                        net.optim_ae.step()

                    net.optim_gen.step()

            if not sv.batch_step % cfg.log_interval == 0:
                continue

            # exponentially decaying noise on autoencoder
            # noise_raius = 0.2(default)
            # noise_anneal = 0.995(default)
            net.ae.noise_radius = net.ae.noise_radius * cfg.noise_anneal

            epoch = sv.epoch_step
            nbatch = sv.batch_step
            niter = sv.global_step

            # Autoencoder
            batch = net.data_eval.next()
            tars, outs = Autoencoder.eval_(cfg, net.ae, batch)
            print_info(sv)
            print_nums(sv, 'AutoEnc', dict(Loss=ae_loss,
                                           Accuracy=ae_acc))
            print_ae_sents(net.vocab, tars, outs, batch.len, cfg.log_nsample)

            # Generator + Discriminator_c
            fake_hidden = Generator.generate_(cfg, net.gen, fixed_noise, False)
            ids_fake_eval, _ = Autoencoder.decode_(cfg, net.ae, fake_hidden,
                                                   net.vocab, False)
            print_info(sv)
            print_nums(sv, 'CodeGAN Loss', dict(D_Total=err_dc_total,
                                                D_Real=err_dc_real,
                                                D_Fake=err_dc_fake,
                                                G=err_g))
            print_gen_sents(net.vocab, ids_fake_eval, cfg.log_nsample)

            # Discriminator_s
            if cfg.with_attn and epoch >= cfg.disc_s_hold:
                print_info(sv)
                loss_ds_total, loss_ds_real, loss_ds_fake = loss_ds
                acc_ds_real, acc_ds_fake = acc_ds
                attns_real, attns_fake = attns
                print_nums(sv, 'SampleGAN Loss', dict(D_Total=loss_ds_total,
                                                      D_Real=loss_ds_real,
                                                      D_Fake=loss_ds_fake,
                                                      G_Dec=loss_dec))
                print_nums(sv, 'SampleGAN Accuracy', dict(D_Real=acc_ds_real,
                                                          D_Fake=acc_ds_fake,
                                                          G_Dec=acc_dec))

                ids_tar = batch.tar.view(cfg.batch_size, -1).data.cpu().numpy()
                attns_fake_r, attns_fake_f = halve_attns(attns_fake)
                print_attns(cfg, net.vocab,
                            dict(Real=(ids_tar, attns_real),
                                 Fake_R=(ids_real, attns_fake_r),
                                 Fake_F=(ids_fake, attns_fake_f)))

            fake_sents = ids_to_sent_for_eval(net.vocab, ids_fake_eval)
            scores = evaluate_sents(test_sents, fake_sents)
            log.info(scores) # NOTE: change later!

            # Autoencoder
            writer.add_scalar('AE/1_AE_loss', ae_loss, niter)
            writer.add_scalar('AE/2_AE_accuracy',  ae_acc, niter)

            # Discriminator_c + Generator
            writer.add_scalar('GAN_C_Loss/1_D_total', err_dc_total, niter)
            writer.add_scalar('GAN_C_Loss/2_D_real', err_dc_real, niter)
            writer.add_scalar('GAN_C_Loss/3_D_fake', err_dc_fake, niter)
            writer.add_scalar('GAN_C_Loss/4_G', err_g, niter)

            # Discriminator_s + Decoder(the other Generator)
            if cfg.with_attn and epoch >= cfg.disc_s_hold:
                writer.add_scalar('GAN_S_Loss/1_D_Total', loss_ds_total, niter)
                writer.add_scalar('GAN_S_Loss/2_D_Real', loss_ds_real, niter)
                writer.add_scalar('GAN_S_Loss/3_D_Fake', loss_ds_fake, niter)
                writer.add_scalar('GAN_S_Loss/4_G_dec', loss_dec, niter)
                writer.add_scalar('GAN_S_Acc/1_D_Real', acc_ds_real, niter)
                writer.add_scalar('GAN_S_Acc/2_D_Fake', acc_ds_fake, niter)

            writer.add_scalar('Eval/1_Bleu', scores['bleu'], niter)
            writer.add_scalar('Eval/2_Meteor', scores['meteor'], niter)
            writer.add_scalar('Eval/3_ExactMatch', scores['em'], niter)
            writer.add_scalar('Eval/4_F1', scores['f1'], niter)

            sv.save()

        # end of epoch ----------------------------
sv.inc_epoch_step()

