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

log = logging.getLogger('main')


class SimpleQAMultiProcessor(LargeFileMultiProcessor):
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

        sents = []
        counter = Counter()
        log.info('\nMerging results from %d files...' % len(processors))
        for results in pool_results:
            sents.extend(results[0])
            counter += results[1]
        return sents, counter

    def process(self):
        results = super(BookCorpusMultiProcessor, self).process()
        log.info('\n' * (self.num_process - 1)) # to prevent dirty print

        sents = []
        counter = Counter()
        log.info('\nMerging the results from multi-processes...')
        for i in tqdm(range(len(results)), total=len(results)):
            sents.extend(results[i][0])
            counter += results[i][1]
        return sents, counter

    def _process_chunk(self, chunk):
        i, start, end = chunk
        chunk_size = end - start
        processed = list()
        counter = Counter()
        tokenizer = self._get_tokenizer(self.tokenizer)

        def process_line(line):
            # replace
            replaces = [("''", '"'), ("``", '"'), ('\\*', '*')]
            for src, dst in replaces:
                line = line.replace(src, dst)
            # tokenize line & count words
            tokens = tokenizer(line.strip())
            if len(tokens) > self.max_len or len(tokens) < self.min_len:
                return None
            return [token.lower() for token in tokens]

        with open(self.file_path, 'r') as f:
            f.seek(start)
            # process multiple chunks simultaneously with progress bar
            text = '[Process #%2d] ' % i
            with tqdm(total=chunk_size, desc=text, position=i) as pbar:
                while f.tell() < end:
                    curr = f.tell()
                    line = f.readline()
                    pbar.update(f.tell() - curr)
                    tokens = process_line(line)
                    if tokens is not None:
                        processed.append(tokens)
                        counter.update(tokens)
        return processed, counter

    def _get_tokenizer(self, tokenizer):
        if tokenizer == "spacy":
            spacy_en = spacy.load('en')
            return lambda s: [tok.text for tok in spacy_en.tokenizer(s)]
        elif tokenizer == "nltk": # NOTE : not working on Python 3.6.x
            return lambda s: [tok for tok in nltk.word_tokenize(s)]


class SimpleQADataset(Dataset):
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
