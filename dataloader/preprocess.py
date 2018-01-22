import numpy as np
import logging
import os

from dataloader.book_corpus import BookCorpusMultiProcessor
from dataloader.vocab import Vocab, GloveMultiProcessor
from utils.utils import StopWatch

log = logging.getLogger('main')


def preprocess_data_vocab(cfg):
    StopWatch.go('Total')
    if (not os.path.exists(cfg.data_filepath)
        or not os.path.exists(cfg.vocab_filepath)
        or cfg.reload_prepro):

        log.info('Start preprocessing data and building vocabulary!')
        if isinstance(cfg.train_filepath, (list, tuple)):
            book_procs = BookCorpusMultiProcessor.from_multiple_files(
                    file_paths=cfg.train_filepath,
                    min_len=cfg.min_len,
                    max_len=cfg.max_len)
            sents, counter = BookCorpusMultiProcessor.multi_process(book_procs)
        else:
            book_procs = BookCorpusMultiProcessor(file_path=cfg.train_filepath,
                                                  min_len=cfg.min_len,
                                                  max_len=cfg.max_len)
            sents, counter = book_procs.process()

        # pretrained embedding initialization if necessary
        if cfg.load_glove:
            print('Loading GloVe pretrained embeddings...')
            glove_processor = GloveMultiProcessor(glove_dir=cfg.glove_dir,
                                                  vector_size=cfg.embed_size)
            word2vec = glove_processor.process()
        else:
            word2vec = None

        vocab = Vocab(counter=counter,
                      max_size=cfg.vocab_size,
                      embed_dim=cfg.embed_size,
                      init_embed=word2vec)

        sents = vocab.numericalize_sents(sents)


        with StopWatch('Saving text'):
            np.savetxt(cfg.data_filepath, sents, fmt="%s")
            log.info("Saved preprocessed data: %s", cfg.data_filepath)
        with StopWatch('Pickling vocab'):
            vocab.pickle(cfg.vocab_filepath)
            log.info("Saved vocabulary: %s" % cfg.vocab_filepath)
    else:
        log.info('Previously processed files will be used!')
        vocab = Vocab.unpickle(cfg.vocab_filepath)
    StopWatch.stop('Total')
    return vocab
