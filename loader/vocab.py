from collections import Counter
import logging
import numpy as np
import os
import pickle
from tqdm import tqdm

from loader.multi_proc import LargeFileMultiProcessor

log = logging.getLogger('main')


class Vocab(object):
    def __init__(self, counter, specials=None, max_size=None, min_freq=None):
        log.info('\nBuilding vocabulary...')
        self.word2idx = dict()
        self.idx2word = list()
        self._embed = None
        self._update_id_attr(specials)

        if specials:
            # update special tokens
            specials_ = {token: idx for idx, token in enumerate(specials)}
            self.word2idx.update(specials_)
            self.idx2word = specials.copy()
            self.specials = specials
        else:
            self.specials = []

        # filter by the minimum frequency
        if min_freq is not None:
            filtered = {k: c for k, c in counter.items() if c > min_freq}
            counter = Counter(filtered)

        # filter by frequency
        if max_size is None:
            words_freq = counter.most_common()
        else:
            words_freq = counter.most_common(max_size - len(self.specials))

        # sort by alphbetical order
        words_freq.sort(key=lambda tup: tup[0])

        # update word2idx & idx2word
        for word, freq in words_freq:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1

        vocab_size = len(self)

    def __len__(self):
        return len(self.word2idx)

    def _update_id_attr(self, specials):
        # make id attributes e.g. PAD_ID, EOS_ID, ...
        specials = {special.strip("<>").upper() + '_ID' : i
                    for i, special in enumerate(specials)}
        self.__dict__.update(specials)

    @property
    def embed(self):
        if self._embed is None:
            raise Exception("Embeddings has not been generated.\n"
                            "Run generated_embedding before call Vocab.embed!")
        else:
            return self._embed

    def generate_embedding(self, embed_dim, init_embed=None):
        # standard gaussian distribution initialization
        self._embed = np.random.normal(size=(len(self), embed_dim))

        if init_embed is not None:
            for word, idx in self.word2idx.items():
                self._embed[idx] = init_embed.get(word, self._embed[idx])
        # embedding of <pad> token should be zero
        if self.idx2word[self.PAD_ID] in self.word2idx.keys():
            self._embed[self.PAD_ID] = 0

    def ids2text_batch(self, ids_batch):
        return list(map(self.ids2text, ids_batch))

    def ids2words_batch(self, ids_batch):
        return list(map(self.ids2words, ids_batch))

    def word2ids_batch(self, word_batch):
        # convert words in sentences to indices
        # sents : [ [tok1, tok2, ... ], [tok1, tok2], ... ]
        return list(map(self.words2ids, word_batch))

    def ids2text(self, ids):
        words = []
        for word in self.ids2words(ids):
            if word == self.idx2word[self.PAD_ID]:
                break
            else:
                words.append(word)
        return ' '.join(words)

    def ids2words(self, ids):
        return [self.idx2word[idx] for idx in ids]

    def words2ids(self, words):
        return [self.word2idx.get(word, self.UNK_ID) for word in words]

    def remove_pads_from_txt(self, txt):
        return txt.replace(self.idx2word[self.PAD_ID], '')

    def remove_after_first_pad(self, txt):
        return txt[:txt.find(self.idx2word[self.PAD_ID])]

    def pickle(self, file_path):
        log.info('Pickling : %s' % file_path)
        with open(file_path, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def unpickle(file_path):
        log.info('Unpickling : %s' % file_path)
        with open(file_path, 'rb') as f:
            return pickle.load(f)
