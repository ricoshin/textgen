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
    def __init__(self, source, target, length, postag=None):
        self.__source = source
        self.__target = target
        self.__length = length
        self.__postag = postag

    @property
    def src(self):
        return self.__source

    @property
    def tar(self):
        return self.__target

    @property
    def len(self):
        return self.__length

    @property
    def pos(self):
        if self.__postag is None:
            raise Exception("POS tage has not been initialzed!")
        else:
            return self.__postag

    def variable(self, volatile=False):
        source = Variable(self.__source, volatile=volatile)
        target = Variable(self.__target, volatile=volatile)
        if self.__postag is not None:
            postag = Variable(self.__postag, volatile=volatile)
        else:
            postag = None
        return Batch(source, target, self.__length, postag)

    def cuda(self, cuda=True):
        if cuda:
            source = self.__source.cuda()
            target = self.__target.cuda()
            if self.__postag is not None:
                postag = self.__postag.cuda()
            else:
                postag = None
        else:
            source = self.__source
            target = self.__target
            if self.__postag is not None:
                postag = self.__postag
            else:
                postag = None
        return Batch(source, target, self.__length, postag)


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
        postag = []
        lengths = []

        for sent, pos in batch:
            # sent and pos length must be the same
            assert(len(sent) == len(pos))
            lengths.append(len(sent))
        batch_max_len = max(lengths)

        # Sort samples in decending order in order to use pack_padded_sequence
        if len(batch) > 1:
            batch, lengths = self._length_sort(batch, lengths)

        for sent, pos in batch:
            # pad & sos/eos
            num_pads = batch_max_len - len(sent)
            pads = [self.vocab.PAD_ID] * num_pads
            src = sent + pads
            tar = sent + [self.vocab.EOS_ID] + pads
            pos = pos + [self.vocab_pos.EOS_ID] + pads # eos : for matching step length

            source.append(src)
            target.append(tar)
            postag.append(pos)

        source = torch.LongTensor(np.array(source))
        target = torch.LongTensor(np.array(target)).view(-1)
        postag = torch.LongTensor(np.array(postag)).view(-1)

        if self.gpu:
            source = source.cuda()
            target = target.cuda()
            postag = postag.cuda()

        return Batch(source, target, lengths, postag)


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
