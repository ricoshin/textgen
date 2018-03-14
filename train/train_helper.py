import logging
import numpy as np
import os

import torch
from torch.autograd import Variable

from utils.utils import to_gpu

log = logging.getLogger('main')


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

def print_ae_tf_sents(vocab, target_ids, output_ids, lengths, nline=5):
    coupled = list(zip(target_ids, output_ids, lengths))
    # shuffle : to prevent always printing the longest ones first
    np.random.shuffle(coupled)
    print_line()
    for i, (tar_ids, out_ids, length) in enumerate(coupled):
        if i > nline - 1: break
        log.info("[X] " + ids_to_sent(vocab, tar_ids, length=length))
        log.info("[Y] " + ids_to_sent(vocab, out_ids, length=length))
        print_line()

def print_ae_tf_sents(vocab, target_ids, output_ids, lengths, nline=5):
    coupled = list(zip(target_ids, output_ids, lengths))
    # shuffle : to prevent always printing the longest ones first
    np.random.shuffle(coupled)
    print_line()
    for i, (tar_ids, out_ids, length) in enumerate(coupled):
        if i > nline - 1: break
        log.info("[X] " + ids_to_sent(vocab, tar_ids, length=length))
        log.info("[Y] " + ids_to_sent(vocab, out_ids, length=length))
        print_line()

def print_ae_fr_sents(vocab, target_ids, output_ids, nline=5):
    coupled = list(zip(target_ids, output_ids))
    # shuffle : to prevent always printing the longest ones first
    np.random.shuffle(coupled)
    print_line()
    for i, (tar_ids, out_ids) in enumerate(coupled):
        if i > nline - 1: break
        log.info("[X] " + ids_to_sent(vocab, tar_ids, no_pad=False))
        log.info("[Y] " + ids_to_sent(vocab, out_ids, no_pad=False))
        print_line()

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

def mark_empty_attn_w(attns, cfg):
    filters = cfg.arch_enc_disc.f
    strides = cfg.arch_enc_disc.s
    assert len(filters) == len(attns)
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
        for j in range(cfg.max_len - left_empty):
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
        #import pdb; pdb.set_trace()
        attns_w = mark_empty_attn_w(attns_w, cfg)
        attns_w, attns_l = batch_first_attns([attns_w, attns_l])
        coupled = list(zip(bat_ids, attns_w, attns_l))
        np.random.shuffle(coupled)
        for i, sent_wise in enumerate(coupled):
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

def mask_sequence_with_n_inf(seqs, seq_lens):
    max_seq_len = seqs.size(1)
    masks = seqs.data.new(*seqs.size()).zero_()
    for mask, seq_len in zip(masks, seq_lens):
        seq_len = seq_len.data[0]
        if seq_len < max_seq_len:
            mask[seq_len:] = float('-inf')
    masks = Variable(masks, requires_grad=False)
    return seqs + masks

def to_one_hot(cfg, indices, num_class):
    size = indices.size()
    dim = len(size)
    indices = torch.unsqueeze(indices.data, dim)
    one_hot = torch.FloatTensor(*size, num_class).zero_()
    if isinstance(indices, Variable):
        one_hot = Variable(one_hot, requires_grad=False)
    if cfg.cuda:
        one_hot = to_gpu(cfg.cuda, one_hot)
    one_hot.scatter_(dim, indices, 1.)
    return one_hot

def compute_cosine_sim(out_word, embedding):
    # compute cosine similarity
    embed = embedding.embed.weight.detach()
    vocab_size, embed_size = embed.size()
    embed = embed.permute(1, 0) # [embed_size, vocab_size]
    out_embed = out_word.view(-1, embed_size) # [bsz(*maxlen), embed_size]
    cos_sim = torch.mm(out_embed, embed) # [bsz(*maxlen), vocab_size]
    cos_sim = cos_sim.view(*out_word.size()[:-1], vocab_size)
    return cos_sim # [bsz, (max_len,) vocab_size]
