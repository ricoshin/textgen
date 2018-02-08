from collections import Counter
import linecache
import logging
import multiprocessing as mp
import numpy as np
import spacy
# import nltk # NOTE not available on python 3.6.x
from tqdm import tqdm

import torch
from torch.autograd import Variable
from torch.utils.data import Dataset

from loader.multi_proc import LargeFileMultiProcessor, LineCounter
from utils.utils import to_gpu

import pdb

log = logging.getLogger('main')


class SimpleQuestionsMultiProcessor(LargeFileMultiProcessor):
    def __init__(self, file_path, num_process=None,
                 min_len=0, max_len=999, tokenizer='spacy'):
        # skip out too short/long sentences
        self.min_len = min_len
        self.max_len = max_len
        self.tokenizer = tokenizer
        super(BookCorpusMultiProcessor, self).__init__(file_path, num_process)

    @classmethod
    def from_multiple_files(cls, file_paths, num_process=None,
                            min_len=0, max_len=999, tokenizer='spacy'):
        if not isinstance(file_paths, (list, tuple)):
            raise TypeError('File_paths must be list or tuple')
        processors = list()
        for file_path in file_paths:
            processors.append(cls(file_path, num_process,
                                  max_len, max_len, tokenizer))
        return processors

    @classmethod
    def multi_process(cls, processors):
        if not isinstance(processors, (list,tuple)):
            raise ValueError('Processors must be list or tuple')

        pool_results = list()
        for processor in processors:
            pool_results.append(processor.process())

        q_sents = []
        a_sents = []
        q_counter = Counter()
        a_counter = Counter()
        log.info('\nMerging results from %d files...' % len(processors))
        for results in pool_results:
            q_sents.extend(results[0])
            a_sents.extend(results[1])
            q_counter += results[2]
            a_counter += results[3]
        return q_sents, a_sents, q_counter, a_counter

    def process(self):
        results = super(BookCorpusMultiProcessor, self).process()
        log.info('\n' * (self.num_process - 1)) # to prevent dirty print

        q_sents = []
        a_sents = []
        q_counter = Counter()
        a_counter = Counter()
        log.info('\nMerging the results from multi-processes...')
        for i in tqdm(range(len(results)), total=len(results)):
            q_sents.extend(results[i][0])
            a_sents.extend(results[i][1])
            q_counter += results[i][2]
            a_counter += results[i][3]
        return q_sents, a_sents, q_counter, a_counter

    def _process_chunk(self, chunk):
        i, start, end = chunk
        chunk_size = end - start
        q_processed = list()
        a_processed = list()
        q_counter = Counter()
        a_counter = Counter()
        tokenizer = self._get_tokenizer(self.tokenizer)

        def process_line(line):
            # remove numbers
            line = line.lstrip("1")
            # split into ques, ans
            line = line.split("\t")
            # replace
            replaces = [("''", '"'), ("``", '"'), ('\\*', '*')]
            tokens = []
            for src, dst in replaces:
                for l in line:
                    l = l.replace(src, dst)
                    # tokenize line & count words
                    token = tokenizer(l.strip())
                    tokens.append([t.lower() for t in token])
            if len(tokens) > self.max_len:
                return None
            #if len(tokens) > self.max_len or len(tokens) < self.min_len:
            #    return None
            return tokens[0], tokens[1]

        with open(self.file_path, 'r') as f:
            f.seek(start)
            # process multiple chunks simultaneously with progress bar
            text = '[Process #%2d] ' % i
            with tqdm(total=chunk_size, desc=text, position=i) as pbar:
                while f.tell() < end:
                    curr = f.tell()
                    line = f.readline()
                    pbar.update(f.tell() - curr)
                    token_q, token_a = process_line(line)
                    if token_q is not None:
                        q_processed.append(token_q)
                        q_counter.update(token_q)
                    if token_a is not None:
                        a_processed.append(token_a)
                        a_counter.update(token_a)
        return q_processed, a_processed,  q_counter, a_counter

    def _get_tokenizer(self, tokenizer):
        if tokenizer == "spacy":
            spacy_en = spacy.load('en')
            return lambda s: [tok.text for tok in spacy_en.tokenizer(s)]
        elif tokenizer == "nltk": # NOTE : not working on Python 3.6.x
            return lambda s: [tok for tok in nltk.word_tokenize(s)]


class SimpleQuestionsDataset(Dataset):
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

class Batch(object):
    def __init__(self, question, question_target, answer, answer_target, length):
        self.__question = question
        self.__question_tar = question_target
        self.__answer = answer
        self.__answer_tar = answer_target
        self.__length = length

    @property
    def q(self):
        return self.__question

    @property
    def q_tar(self):
        return self.__question_tar

    @property
    def a(self):
        return self.__answer

    @property
    def a_tar(self):
        return self.__answer_tar

    @property
    def len(self):
        return self.__length

    def variable(self, volatile=False):
        question = Variable(self.__question, volatile=volatile)
        question_target = Variable(self.__question_tar, volatile=volatile)
        answer = Variable(self.__answer, volatile=volatile)
        answer_target = Variable(self.__answer_tar, volatile=volatile)
        return Batch(question, question_target, answer, answer_target, self.__length)

    def cuda(self, cuda=True):
        if cuda:
            question = self.__question.cuda()
            question_target = self.__question_tar.cuda()
            answer = self.__answer.cuda()
            answer_target = self.__answer_tar.cuda()
        else:
            question = self.__question
            question_target = self.__question_tar
            question = self.__answer
            answer_target = self.__answer_tar
        return Batch(question, question_target, answer, answer_target, self.__length)


class BatchingDataset(object):
    def __init__(self, vocab, gpu=False):
        self.gpu = gpu
        self.pad_id = vocab.PAD_ID
        self.sos_id = vocab.SOS_ID
        self.eos_id = vocab.EOS_ID
        self.split_id = vocab.SPLIT_ID

    def __call__(self, sample_list):
        return self.process(sample_list)

    def process(self, sample_list):
        question = []
        question_target = []
        answer = []
        answer_target = []
        pdb.set_trace()
        for sent in sample_list:
            split_pos = sent.index(self.split_id)
            question.append(sent[:split_pos])
            answer.append(sent[split_pos+1:])

        if len(question) != len(answer): # error handling
            print('length of q and a is not eqal')
            print('q len: ', str(len(question)))
            print('a len: ', str(len(answer)))
            pdb.set_trace()
        lengths = [(len(sent) + 1) for sent in question] # +1: sos/eos
        lengths_temp = lengths
        max_len = max(lengths)

        # Sort samples in decending order in order to use pack_padded_sequence
        if len(question) > 1:
            question, lengths = self._length_sort(question, lengths)
            answer, lengths_temp = self._length_sort(answer, lengths_temp)

        def sort_and_pad(lists):
            out = []
            out_tgt = []
            for sent in lists:
                # pad & sos/eos
                num_pads = max_len - len(sent) - 1
                pads = [self.pad_id] * num_pads
                x = [self.sos_id] + sent + pads
                y = sent + [self.eos_id] + pads
                out.append(x)
                out_tgt.append(y)
            out = torch.LongTensor(np.array(out))
            out_tgt = torch.LongTensor(np.array(out_tgt)).view(-1)
            return out, out_tgt
        question, question_target = sort_and_pad(question)
        answer, answer_target = sort_and_pad(answer)

        if self.gpu:
            question = question.cuda()
            question_target = question_target.cuda()
            answer = answer.cuda()
            answer_taget = answer_target.cuda()

        return Batch(question, question_target, answer, answer_target, lengths)

    def _length_sort(self, items, lengths, descending=True):
        items = list(zip(items, lengths))
        items.sort(key=lambda x: x[1], reverse=True)
        items, lengths = zip(*items)
        return list(items), list(lengths)


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
