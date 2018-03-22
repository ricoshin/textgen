from copy import deepcopy
import collections
import linecache
import logging
import multiprocessing as mp
import numpy as np
import pickle
from tqdm import tqdm

import torch
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataloader import DataLoaderIter

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


class BatchCollator(object):
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
            batch_max_len = self.cfg.max_len #max(lengths) #NOTE CNN/RNN

        for tokens in batch:
            # pad & sos/eos
            num_pads = batch_max_len - len(tokens)
            pads = [self.vocab.PAD_ID] * num_pads
            x = tokens + pads
            y = tokens + pads
            #y = tokens + [self.vocab.EOS_ID] + pads

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


class POSBatchCollator(BatchCollator):
    def __init__(self, cfg, vocab, vocab_pos, gpu=False):
        super(POSBatchCollator, self).__init__(cfg, vocab, gpu)
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
        batch_max_len = self.cfg.max_len #max(lengths) #NOTE CNN/RNN

        # Sort samples in decending order in order to use pack_padded_sequence
        if len(batch) > 1:
            batch, lengths = self._length_sort(batch, lengths)

        for sent, tag in batch:
            # pad & sos/eos
            num_pads = batch_max_len - len(sent)
            pads = [self.vocab.PAD_ID] * num_pads
            src = sent + pads
            src_tag = tag + pads
            tar = sent + pads
            #tar = sent + [self.vocab.EOS_ID] + pads
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


class Batch(object):
    def __init__(self, source, target, length):
        self.src = source
        self.tar = target
        self.len = length

    @property
    def max_len(self):
        return max(self.len)

    def variable(self, volatile=False):
        src = Variable(self.src, volatile=volatile)
        tar = Variable(self.tar, volatile=volatile)
        return Batch(src, tar, self.len)

    def cuda(self, cuda=True):
        if cuda:
            src = self.src.cuda()
            tar = self.tar.cuda()
        else:
            src = self.src
            tar = self.tar
        return Batch(src, tar, self.len)


class BatchTag(Batch):
    def __init__(self, source, target, length, source_tag, target_tag):
        super(BatchTag, self).__init__(source, target, length)
        self.__src_tag = source_tag
        self.__tar_tag = target_tag

    @property
    def max_len(self):
        return max(self.len)

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


class MyDataLoader(DataLoader):
    """In order to return MyDataLoaderIter when iter(MyDataLoader) called"""
    def __iter__(self):
        return MyDataLoaderIter(self)


class MyDataLoaderIter(DataLoaderIter):
    """Class method overriding for pickling DatatLoaderIter"""
    def __init__(self, *inputs):
        super(MyDataLoaderIter, self).__init__(*inputs)

    def __del__(self):
        if hasattr(self, 'num_workers'):
            if self.num_workers > 0:
                self._shutdown_workers()

    def __getstate__(self):
        # log.debug("pickling")
        state_list = ['sample_iter', 'rcvd_idx', 'reorder_dict',
                      'batches_outstanding']
        state = dict()
        for key, value in self.__dict__.items():
            if key in state_list:
                if key == 'sample_iter':
                    # generator to list (generator can't be pickled)
                    value = list(value)
                state.update({key:value})
        return state

    def __setstate__(self, state):
        for key, value in state.items():
            if key == 'sample_iter':
                # list to generator
                value = (val for val in value)
            self.__dict__.update({key:value})


class DataScheduler(object):
    def __init__(self, cfg, dataloader, volatile=False):
        self._dataloader = dataloader
        self._batch_iter = iter(self._dataloader)
        self._batch = None # initial value
        self._cuda = cfg.cuda
        self._volatile = volatile
        self.step = Step(len(dataloader), cfg.epochs)

    @property
    def batch(self):
        return self._batch.variable(self._volatile).cuda(self._cuda)

    def __len__(self):
        return len(self._dataloader)

    def __getstate__(self):
        # pickle only selected states for memory & time cost
        state_list = ['_batch_iter', 'step']  #NOTE : use __setattr__
        return {state: self.__dict__[state] for state in state_list}

    def __setstate__(self, state):
        self.__dict__ = state

    def reset(self):
        self._batch_iter = iter(self._dataloader)

    def next(self):
        self.step.increase()
        self._batch = next(self._batch_iter, None)
        if self._batch is None:
            self.reset()
            self._batch = next(self._batch_iter)
        return self.batch

    def save_as_pickle(self, file_path):
        with open(file_path, 'wb') as f:
            pickle.dump(self, f)

    def load_from_pickle(self, file_path):
        with open(file_path, 'rb') as f:
            loaded = pickle.load(f)
            self._update_recursively(self, loaded)

    def _update_recursively(self, tar, src):
        # Updates state dict recursively preserving original attributes
        tar_dict = tar.__dict__
        src_dict = src.__dict__
        for src_key, src_val in src_dict.items():
            tar_val = tar_dict.get(src_key)
            if hasattr(tar_val, '__dict__'):
                self._update_recursively(tar_dict[src_key], src_dict[src_key])
            else:
                tar_dict[src_key] = src_val


class Step(object):
    def __init__(self, num_batch, num_epoch):
        self.batch = 0
        self.epoch = 0

        self.batch_max = num_batch  # 1000
        self.epoch_max = num_epoch  # 15
        self.total_max = num_batch*num_epoch

    @property
    def total(self):
        return self.batch_max*self.epoch + self.batch

    def increase(self):
        self.batch += 1

        if self.batch >= self.batch_max:
            self.batch = 0
            self.epoch += 1

    def is_end_of_step(self):
        return self.epoch >= self.epoch_max

    def __str__(self):
        return "Epoch : %d/%d | Batches : %d/%d | Total : %d/%d" % (
            self.epoch, self.epoch_max, self.batch, self.batch_max,
            self.total, self.total_max)
