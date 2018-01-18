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
from evaluate_nltk import truncate
from generator import Generator
from sample_disc import SampleDiscriminator

log = logging.getLogger('main')

"""
codes originally from ARAE
some parts are modified
"""
from utils_kenlm import train_ngram_lm, get_ppl

def train_lm(eval_data, gen_data, vocab, save_path, n):
    # ppl = train_lm(eval_data=test_sents, gen_data = fake_sent,
    #     save_path = "output/niter{}_lm_generation".format(niter)
    #     vocabe = net.vocab)
    # input : test dataset
    #kenlm_path = '/home/jwy/venv/env36/lib/python3.6/site-packages/kenlm'
    kenlm_path = '/home/jwy/venv/env36/lib/python3.5/site-packages/kenlm'
    #processing
    eval_sents = [truncate(s) for s in eval_data]
    gen_sents = [truncate(s) for s in gen_data]

    # write generated sentences to text file
    with open(save_path+".txt", "w") as f:
        # laplacian smoothing
        for word in vocab.word2idx.keys():
            if word == '<unk>' or word == '<eos>' or word == '<pad>':
                continue
            f.write(word+"\n")
        for sent in gen_sents:
            chars = " ".join(sent)
            f.write(chars+"\n")

    # train language model on generated examples
    lm = train_ngram_lm(kenlm_path=kenlm_path,
                        data_path=save_path+".txt",
                        output_path=save_path+".arpa",
                        N=n)
    # evaluate
    ppl = get_ppl(lm, eval_sents)
    return ppl

"""
codes originall from ARAE
end here
"""


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
    print_info(sv)
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

def mark_empty_attn(attns, max_len):
    filter_n_stride = [(3,1), (3,2), (3,2), (4,1)]
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
        # layer level
        if i == 0:
            prev_stride = 1
        else:
            prev_stride = strides[i-1]
        left_empty += (filters[i] // 2) * prev_stride
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

def print_attns(cfg, vocab, real_ids, fake_ids, real_attns, fake_attns):
    real_attns[0] = mark_empty_attn(real_attns[0], cfg.max_len + 1)
    fake_attns[0] = mark_empty_attn(fake_attns[0], cfg.max_len + 1)
    real_attns = batch_first_attns(real_attns)
    fake_attns = batch_first_attns(fake_attns)
    # len(real_attns) : batch_size
    # real_attns[0] : [array(attn_1[0]), array(attn_2[0], array(attn_3[0]))
    def print_aligned(bat_ids, bat_attns):
        attns_w, attns_l = bat_attns
        for i, sent_wise in enumerate(zip(bat_ids, attns_w, attns_l)):
            ids, attns_w, attns_l = sent_wise
            if i > cfg.log_nsample - 1: break
            # ids = pad_after_eos(vocab, ids) # redundant for real_ids
            words = [vocab.idx2word[idx] for idx in ids]
            word_str, attn_str_list = align_word_attn(words, attns_w, attns_l)
            for attn_str in reversed(attn_str_list): # from topmost attn layer
                log.info(attn_str)
            log.info(word_str)
            print_line()

    print_line()
    log.info('Attention on real samples')
    print_line()
    print_aligned(real_ids, real_attns)
    log.info('Attention on fake samples')
    print_line()
    print_aligned(fake_ids, fake_attns)
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
        return torch.cat([tensor, pads], dim=1)
    else:
        return tensor

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
                    real_code = Autoencoder.encode_(cfg, net.ae, batch)
                    fake_code = Generator.generate_(cfg, net.gen, None, False)
                    errs_d_c= CodeDiscriminator.train_(
                        cfg, net.disc_c, net.ae, real_code, fake_code)
                    err_d_c, err_d_c_real, err_d_c_fake = errs_d_c

                    # train SampleDiscriminator
                    real_ids, real_outs = \
                        Autoencoder.decode_(cfg, net.ae, real_code, net.vocab)
                    fake_ids, fake_outs = \
                        Autoencoder.decode_(cfg, net.ae, fake_code, net.vocab)

                    if cfg.with_attn and sv.epoch_step >= cfg.disc_s_hold:
                        if cfg.disc_s_in == 'embed':
                            # "real" fake
                            fake_outs = torch.cat([real_outs, fake_outs], dim=0)
                            # "real" real
                            real_outs = append_pads(cfg, batch.src, net.vocab)

                        errs_d_s, attns = SampleDiscriminator.train_(
                            cfg, net.disc_s, real_outs, fake_outs)
                        err_d_s, err_d_s_real, err_d_s_fake = errs_d_s

                    net.optim_ae.step()
                    net.optim_disc_c.step()
                    if cfg.with_attn:
                        net.optim_disc_s.step()

                # train generator(with disc_c) / decoder(with disc_s)
                for i in range(cfg.niters_gan_g): # default: 1
                    err_g, fake_code = Generator.train_(cfg, net.gen, net.ae,
                                                        net.disc_c)
                    net.optim_gen.step()
                    if cfg.with_attn and sv.epoch_step >= cfg.disc_s_hold:
                        err_dec = Autoencoder.decoder_train_(
                            cfg, net.ae, net.disc_s, fake_code, net.vocab)
                        net.optim_ae.step()

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
            tars, outs = Autoencoder.eval_(cfg, net.ae, batch)
            print_nums(sv, 'AutoEnc', dict(Loss=ae_loss,
                                           Accuracy=ae_acc))
            print_ae_sents(net.vocab, tars, outs, batch.len, cfg.log_nsample)

            # Generator + Discriminator_c
            fake_hidden = Generator.generate_(cfg, net.gen, fixed_noise, False)
            fake_ids, _ = Autoencoder.decode_(cfg, net.ae, fake_hidden,
                                              net.vocab, False)
            print_nums(sv, 'CodeGAN', dict(Loss_D_Total=err_d_c,
                                           Loss_D_Real=err_d_c_real,
                                           Loss_D_Fake=err_d_c_fake,
                                           Loss_G=err_g))
            print_gen_sents(net.vocab, fake_ids, cfg.log_nsample)

            # Discriminator_s
            if cfg.with_attn and epoch >= cfg.disc_s_hold:
                print_nums(sv, 'SampleGAN', dict(Loss_D_Total=err_d_s,
                                                 Loss_D_Real=err_d_s_real,
                                                 Loss_D_Fake=err_d_s_fake,
                                                 Loss_Dec=err_dec))
                print_attns(cfg, net.vocab, real_ids, fake_ids, *attns)

            fake_sents = ids_to_sent_for_eval(net.vocab, fake_ids)
            scores = evaluate_sents(test_sents, fake_sents)
            log.info(scores) # NOTE: change later!

            ### added by JWY
            ppl = train_lm(eval_data=test_sents, gen_data = fake_sents,
                vocab = net.vocab,
                save_path = "out/{}/niter{}_lm_generation".format(sv.cfg.name, niter),
                n = cfg.N)
            print("Perplexity {}".format(ppl))
            writer.add_scalar('Reverse_Perplexity', ppl, niter)
            ### end

            # Autoencoder
            writer.add_scalar('AE/1_AE_loss', ae_loss, niter)
            writer.add_scalar('AE/2_AE_accuracy',  ae_acc, niter)

            # Discriminator_c + Generator
            writer.add_scalar('GAN_c/1_Disc_loss_total', err_d_c, niter)
            writer.add_scalar('GAN_c/2_Disc_loss_real', err_d_c_real, niter)
            writer.add_scalar('GAN_c/3_Disc_loss_fake', err_d_c_fake, niter)
            writer.add_scalar('GAN_c/4_Gen_loss', err_g, niter)

            # Discriminator_s + Decoder(the other Generator)
            if cfg.with_attn and epoch >= cfg.disc_s_hold:
                writer.add_scalar('GAN_s/1_Disc_loss', err_d_s, niter)
                writer.add_scalar('GAN_s/2_Disc_loss_real', err_d_s_real, niter)
                writer.add_scalar('GAN_s/3_Disc_loss_fake', err_d_s_fake, niter)
                writer.add_scalar('GAN_s/4_Dec_loss', err_dec, niter)

            writer.add_scalar('Eval/1_Bleu', scores['bleu'], niter)
            writer.add_scalar('Eval/2_Meteor', scores['meteor'], niter)
            writer.add_scalar('Eval/3_ExactMatch', scores['em'], niter)
            writer.add_scalar('Eval/4_F1', scores['f1'], niter)

            sv.save()

        # end of epoch ----------------------------
        sv.inc_epoch_step()
