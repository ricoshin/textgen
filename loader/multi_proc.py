from collections import Counter
import logging
import multiprocessing as mp
import os

import spacy
# import nltk # NOTE not available on python 3.6.x
from tqdm import tqdm

log = logging.getLogger('main')

class LargeFileMultiProcessor(object):
    def __init__(self, file_path, num_process=None, verbose=True):
        self.file_path = file_path
        self.file_size = os.path.getsize(file_path)
        self.verbose = verbose

        # use all cpu cores if not specifed
        if num_process is None:
            self.num_process = mp.cpu_count()
        self.chunk_size_min = int(self.file_size/self.num_process)

    def process(self):
        if self.verbose:
            log.info('')
            log.info('Large file multiprocessor launched.')
            log.info('File : %s' % self.file_path)
            log.info('Size of file : %s' % self.file_size)
            log.info('Number of processes : %s' % self.num_process)

        chunks = []
        with open(self.file_path, "rb") as f:
            for i in range(self.num_process):
                start = f.tell()
                f.seek(self.chunk_size_min, os.SEEK_CUR)
                if f.readline(): # go to the end of the line
                    end = f.tell()
                else:
                    end = f.seek(0, os.SEEK_END)
                chunks.append([i, start, end])

        if self.verbose:
            log.info('Preparing for multiprocessing...')
        pool = mp.Pool(processes=self.num_process)
        pool_results = pool.map(self._process_chunk, chunks)

        pool.close() # no more tasks
        pool.join() # wrap up current tasks
        return pool_results

    def _process_chunk(self, chunks):
        raise NotImplementedError


class LineCounter(LargeFileMultiProcessor):
    @classmethod
    def count(cls, file_path, num_process=None):
        processor = cls(file_path, num_process, verbose=False)
        pool_results =  processor.process()
        return sum(pool_results)

    def _blocks(self, f, start, end, read_size=64*1024): # 65536
        chunk_size = end - start
        _break = False
        while not _break:
            if _break: break
            if (f.tell() + read_size) > (start + chunk_size):
                read_size = int(start + chunk_size - f.tell())
                _break = True
            yield f.read(read_size)

    def _process_chunk(self, chunk):
        i, start, end = chunk
        num_line = 0
        with open(self.file_path, "r") as f:
            f.seek(start)
            for i, block  in enumerate(self._blocks(f, start, end)):
                num_line +=  block.count('\n')
        return num_line


class CorpusMultiProcessor(LargeFileMultiProcessor):
    def __init__(self, file_path, num_process=None, min_len=1, max_len=999,
                 lower=True, tokenizer='spacy', pos_tagging=False):
        # skip out too short/long sentences
        self.min_len = min_len
        self.max_len = max_len
        self.lower = lower
        self.tokenizer = tokenizer
        super(CorpusMultiProcessor, self).__init__(file_path, num_process)

    @classmethod
    def from_multiple_files(cls, file_paths, num_process=None,
                            min_len=1, max_len=999, tokenizer='spacy'):
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
        results = super(CorpusMultiProcessor, self).process()
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
            if len(tokens) < self.min_len:
                return None
            if self.lower:
                return [token.lower() for token in tokens[:self.max_len]]
            else:
                return tokens

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
        elif tokenizer == "split":
            return lambda s: s.split()
        else:
            raise Exception("Unknown tokenizer!")


class CorpusTagMultiProcessor(LargeFileMultiProcessor):
    def __init__(self, file_path, num_process=None, min_len=1, max_len=999,
                 lower=False):
        # skip out too short/long sentences
        self.min_len = min_len
        self.max_len = max_len
        self.lower = lower
        super(CorpusTagMultiProcessor, self).__init__(file_path, num_process)

    @classmethod
    def from_multiple_files(cls, file_paths, num_process=None,
                            min_len=1, max_len=999):
        if not isinstance(file_paths, (list, tuple)):
            raise TypeError('File_paths must be list or tuple')
        processors = list()
        for file_path in file_paths:
            processors.append(cls(file_path, num_process,
                                  max_len, max_len))
        return processors

    @classmethod
    def multi_process(cls, processors):
        if not isinstance(processors, (list,tuple)):
            raise ValueError('Processors must be list or tuple')

        pool_results = list()
        for processor in processors:
            pool_results.append(processor.process())

        tokens = []
        tags = []
        token_cnt = Counter()
        tag_cnt = Counter()

        log.info('\nMerging results from %d files...' % len(processors))
        for results in pool_results:
            tokens.extend(results[0])
            tags.extend(results[1])
            token_cnt += results[2]
            tag_cnt += results[3]

        return tokens, tags, token_cnt, tag_cnt

    def process(self):
        results = super(CorpusTagMultiProcessor, self).process()
        log.info('\n' * (self.num_process - 1)) # to prevent dirty print

        tokens = []
        tags = []
        token_cnt = Counter()
        tag_cnt = Counter()

        log.info('\nMerging the results from multi-processes...')
        for i in tqdm(range(len(results)), total=len(results)):
            tokens.extend(results[i][0])
            tags.extend(results[i][1])
            token_cnt += results[i][2]
            tag_cnt += results[i][3]

        return tokens, tags, token_cnt, tag_cnt

    def _process_chunk(self, chunk):
        i, start, end = chunk
        chunk_size = end - start
        token_list = list()
        tag_list = list()
        token_cnt = Counter()
        tag_cnt = Counter()
        nlp = spacy.load('en_core_web_sm')

        def process_line(line):
            # replace
            replaces = [("''", '"'), ("``", '"'), ('\\*', '*')]
            for src, dst in replaces:
                line = line.replace(src, dst)
            # tokenization & tagging
            doc = nlp(line.strip())
            tokens = [token.text for token in doc]
            tags = [token.tag_ for token in doc]
            assert(len(tokens) == len(tags))
            # lower case
            if self.lower:
                tokens = [token.lower() for token in tokens]
            # min/max length filtering
            if len(tokens) > self.max_len or len(tokens) < self.min_len:
                return None
            else:
                return tokens, tags

        with open(self.file_path, 'r') as f:
            f.seek(start)
            # process multiple chunks simultaneously with progress bar
            text = '[Process #%2d] ' % i
            with tqdm(total=chunk_size, desc=text, position=i) as pbar:
                while f.tell() < end:
                    curr = f.tell()
                    line = f.readline()
                    pbar.update(f.tell() - curr)
                    processed = process_line(line)
                    if processed is not None:
                        tokens, tags = processed
                        token_list.append(tokens)
                        tag_list.append(tags)
                        token_cnt.update(tokens)
                        tag_cnt.update(tags)

        results = [token_list, tag_list, token_cnt, tag_cnt]
        return results


class GloveMultiProcessor(LargeFileMultiProcessor):
    def __init__(self, glove_dir, vector_size, num_process=None):
        glove_files = {
            50: 'glove.6B.50d.txt',
            100: 'glove.6B.100d.txt',
            200: 'glove.6B.200d.txt',
            300: 'glove.840B.300d.txt',
        }
        file_path = os.path.join(glove_dir, glove_files[vector_size])
        self.vector_size = vector_size # may not be used
        super(GloveMultiProcessor, self).__init__(file_path, num_process)

    def process(self):
        results = super(GloveMultiProcessor, self).process()
        log.info('\n' * (self.num_process - 1)) # to prevent dirty print

        word2vec = dict()
        log.info('Merging the results from multi-processes...')
        for i in tqdm(range(len(results)), total=len(results)):
            word2vec.update(results[i])
        return word2vec

    def _process_chunk(self, chunk):
        i, start, end = chunk
        chunk_size = end - start
        word2vec = dict()

        def process_line(line):
            split_line = line.strip().split()
            word = ' '.join(split_line[:-self.vector_size])
            vector = [float(x) for x in split_line[-self.vector_size:]]
            word2vec[word] = vector

        with open(self.file_path, 'r') as f:
            f.seek(start)
            # process multiple chunks simultaneously with progress bar
            text = '[Process #%2d] ' % i
            with tqdm(total=chunk_size, desc=text, position=i) as pbar:
                while f.tell() < end:
                    curr = f.tell()
                    line = f.readline()
                    pbar.update(f.tell() - curr)
                    process_line(line)
        return word2vec
