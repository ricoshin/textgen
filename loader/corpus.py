import linecache
import logging
import multiprocessing as mp
import numpy as np

import torch
from torch.autograd import Variable
from torch.utils.data import Dataset

from loader.multi_proc import LargeFileMultiProcessor, LineCounter
from utils.utils import to_gpu

log = logging.getLogger('main')


class CorpusDataset(Dataset):
    def __init__(self, file_path, vocab=None):
        self.file_path = file_path
        self.getline_fn = linecache.getline

    def __len__(self):
        return LineCounter.count(self.file_path)

    def __getitem__(self, idx):
        line = self.getline_fn(self.file_path, idx+1)
        #tokens = [int(x.strip(',')) for x in line.strip('[]\n').split()]
        #source = tokens
        return [int(x.strip(',')) for x in line.strip('[]\n').split()]


class CorpusPOSDataset(Dataset):
    def __init__(self, sent_path, tag_path, vocab=None):
        self.sent_path = sent_path
        self.tag_path = tag_path
        self.getline_fn = linecache.getline

    def __len__(self):
        return LineCounter.count(self.sent_path)

    def __getitem__(self, idx):
        sent = self.getline_fn(self.sent_path, idx+1)
        tag = self.getline_fn(self.tag_path, idx+1)
        sent = [int(x.strip(',')) for x in sent.strip('[]\n').split()]
        tag = [int(x.strip(',')) for x in tag.strip('[]\n').split()]
        return sent, tag


class Batch(object):
    def __init__(self, source, target, length):
        self.__src = source
        self.__tar = target
        self.__len = length

    @property
    def src(self):
        return self.__src

    @property
    def tar(self):
        return self.__tar

    @property
    def len(self):
        return self.__len

    def variable(self, volatile=False):
        src = Variable(self.__src, volatile=volatile)
        tar = Variable(self.__tar, volatile=volatile)
        return Batch(src, tar, self.__len)

    def cuda(self, cuda=True):
        if cuda:
            src = self.__src.cuda()
            tar = self.__tar.cuda()
        else:
            src = self.__src
            tar = self.__tar
        return Batch(src, tar, self.__len)


class BatchTag(Batch):
    def __init__(self, source, target, length, source_tag, target_tag):
        super(BatchTag, self).__init__(source, target, length)
        self.__src_tag = source_tag
        self.__tar_tag = target_tag

    @property
    def src_tag(self):
        if self.__src_tag is None:
            raise Exception("POS tag has not been initialzed!")
        else:
            return self.__src_tag

    @property
    def tar_tag(self):
        if self.__tar_tag is None:
            raise Exception("POS tag has not been initialzed!")
        else:
            return self.__tar_tag

    def variable(self, volatile=False):
        batch = super(BatchTag, self).variable(volatile)
        src_tag = Variable(self.__src_tag, volatile=volatile)
        tar_tag = Variable(self.__tar_tag, volatile=volatile)
        return BatchTag(batch.src, batch.tar, batch.len, src_tag, tar_tag)

    def cuda(self, cuda=True):
        batch = super(BatchTag, self).cuda(cuda)
        src_tag = self.__src_tag.cuda()
        tar_tag = self.__tar_tag.cuda()
        return BatchTag(batch.src, batch.tar, batch.len, src_tag, tar_tag)


class BatchingDataset(object):
    def __init__(self, cfg, vocab, gpu=False):
        self.cfg = cfg
        self.gpu = gpu
        self.vocab = vocab

    def __call__(self, batch):
        return self.process(batch)

    def process(self, batch):
        source = []
        target = []
        lengths = [(len(sample)) for sample in batch]
        batch_max_len = max(lengths)

        # Sort samples in decending order in order to use pack_padded_sequence
        if len(batch) > 1:
            batch, lengths = self._length_sort(batch, lengths)

        for tokens in batch:
            # pad & sos/eos
            num_pads = batch_max_len - len(tokens)
            pads = [self.vocab.PAD_ID] * num_pads
            x = tokens + pads
            y = tokens + [self.vocab.EOS_ID] + pads

            source.append(x)
            target.append(y)

        source = torch.LongTensor(np.array(source))
        target = torch.LongTensor(np.array(target)).view(-1)

        if self.gpu:
            source = source.cuda()
            target = target.cuda()

        return Batch(source, target, lengths)

    def _length_sort(self, items, lengths, descending=True):
        items = list(zip(items, lengths))
        items.sort(key=lambda x: x[1], reverse=True)
        items, lengths = zip(*items)
        return list(items), list(lengths)


class BatchingPOSDataset(BatchingDataset):
    def __init__(self, cfg, vocab, vocab_pos, gpu=False):
        super(BatchingPOSDataset, self).__init__(cfg, vocab, gpu)
        self.vocab_pos = vocab_pos

    def process(self, batch):
        source = []
        target = []
        lengths = []
        source_tag = []
        target_tag = []

        for sent, tag in batch:
            # sent and pos length must be the same
            assert(len(sent) == len(tag))
            lengths.append(len(sent))
        batch_max_len = max(lengths)

        # Sort samples in decending order in order to use pack_padded_sequence
        if len(batch) > 1:
            batch, lengths = self._length_sort(batch, lengths)

        for sent, tag in batch:
            # pad & sos/eos
            num_pads = batch_max_len - len(sent)
            pads = [self.vocab.PAD_ID] * num_pads
            src = sent + pads
            src_tag = tag + pads
            tar = sent + [self.vocab.EOS_ID] + pads
            tar_tag = tag + [self.vocab_pos.EOS_ID] + pads

            source.append(src)
            target.append(tar)
            source_tag.append(src_tag)
            target_tag.append(tar_tag)

        source = torch.LongTensor(np.array(source))
        target = torch.LongTensor(np.array(target)).view(-1)
        source_tag = torch.LongTensor(np.array(source_tag))
        target_tag = torch.LongTensor(np.array(target_tag)).view(-1)

        if self.gpu:
            source = source.cuda()
            target = target.cuda()
            source_tag = source_tag.cuda()
            target_tag = target_tag.cuda()

        return BatchTag(source, target, lengths, source_tag, target_tag)


class BatchIterator(object):
    def __init__(self, dataloader, cuda, volatile=False):
        self.__dataloader = dataloader
        self.__batch_iter = iter(self.__dataloader)
        self.__batch = None # initial value
        self.__cuda = cuda
        self.__volatile = volatile

    def __len__(self):
        return len(self.__dataloader)

    @property
    def batch(self):
        return self.__batch.variable(self.__volatile).cuda(self.__cuda)

    def reset(self):
        self.__batch_iter = iter(self.__dataloader)

    def next(self):
        self.__batch = next(self.__batch_iter, None)
        if self.__batch is None:
            self.reset()
            self.__batch = next(self.__batch_iter)
        return self.__batch.variable(self.__volatile).cuda(self.__cuda)
