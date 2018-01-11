from collections import Counter
import logging
import numpy as np
import os
import pickle
from tqdm import tqdm

from multi_proc import LargeFileMultiProcessor

log = logging.getLogger('main')


class Vocab(object):
    # static variables
    PAD_ID = 0
    SOS_ID = 1
    EOS_ID = 2
    UNK_ID = 3
    SPECIALS = ['<pad>', '<sos>', '<eos>', '<unk>']

    def __init__(self, counter, max_size, min_freq=None, embed_dim=300,
                 init_embed=None):

        log.info('\nBuilding vocabulary...')
        self.word2idx = dict()
        self.idx2word = list()

        # update special tokens
        specials_ = {token: idx for idx, token in enumerate(Vocab.SPECIALS)}
        self.word2idx.update(specials_)
        self.idx2word = Vocab.SPECIALS.copy()
        self.specials = Vocab.SPECIALS

        # filter by the minimum frequency
        if min_freq is not None:
            filtered = {k: c for k, c in counter.items() if c > min_freq}
            counter = Counter(filtered)

        # filter by frequency
        words_and_freq = counter.most_common(max_size - len(Vocab.SPECIALS))

        # sort by alphbetical order
        words_and_freq.sort(key=lambda tup: tup[0])

        # update word2idx & idx2word
        for word, freq in words_and_freq:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1

        vocab_size = len(self)
        # standard gaussian distribution initialization
        self.embed_mat = np.random.normal(size=(vocab_size, embed_dim))

        if init_embed is not None:
            for word, idx in self.word2idx.items():
                self.embed_mat[idx] = init_embed.get(word, self.embed_mat[idx])
        # embedding of <pad> token should be zero
        self.embed_mat[self.word2idx['<pad>']] = 0

    def __len__(self):
        return len(self.word2idx)

    def pickle(self, file_path):
        log.info('Pickling : %s' % file_path)
        with open(file_path, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def unpickle(file_path):
        log.info('Unpickling : %s' % file_path)
        with open(file_path, 'rb') as f:
            return pickle.load(f)

    def numericalize_sents(self, sents):
        # convert words in sentences to indices
        # sents : [ [tok1, tok2, ... ], [tok1, tok2], ... ]
        result = list()
        unk_id = self.word2idx['<unk>']
        log.info('\nNumericalizing tokenized sents...')
        for sent in tqdm(sents, total=len(sents)):
            result.append([self.word2idx.get(token, unk_id) for token in sent])
        return result


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
