import numpy as np
import logging
import os

from loader.multi_proc import (CorpusMultiProcessor, CorpusTagMultiProcessor,
                               GloveMultiProcessor)
from loader.vocab import Vocab
from utils.utils import StopWatch

log = logging.getLogger('main')


def process_main_corpus(cfg):
    StopWatch.go('Total')
    if (not os.path.exists(cfg.processed_train_path) or
        not os.path.exists(cfg.processed_test_path) or
        not os.path.exists(cfg.processed_vocab_path) or
        cfg.reload_prepro):

        log.info('Start preprocessing data and building vocabulary!')
        if isinstance(cfg.corpus_train_path, (list, tuple)):
            train_proc = CorpusMultiProcessor.from_multiple_files(
                file_paths=cfg.corpus_train_path,
                min_len=cfg.min_len,
                max_len=cfg.max_len - 1) # NOTE considering <eos>
            train_corp, counter = CorpusMultiProcessor.multi_process(train_proc)
        else:
            train_proc = CorpusMultiProcessor(file_path=cfg.corpus_train_path,
                                              min_len=cfg.min_len,
                                              max_len=cfg.max_len - 1)
            train_corp, counter = train_proc.process()

        test_proc = CorpusMultiProcessor(file_path=cfg.corpus_test_path,
                                         min_len=cfg.min_len,
                                         max_len=cfg.max_len - 1)
        test_corp, _ = test_proc.process()
        # pretrained embedding initialization if necessary
        if cfg.load_glove:
            print('Loading GloVe pretrained embeddings...')
            glove_proc = GloveMultiProcessor(glove_dir=cfg.glove_dir,
                                             vector_size=cfg.embed_size_w)
            word2vec = glove_proc.process()
        else:
            word2vec = None

        vocab = Vocab(counter=counter,
                      embed_size=cfg.embed_size_w,
                      embed_init=word2vec,
                      max_size=cfg.vocab_size_w,
                      specials=['<pad>', '<sos>', '<eos>', '<unk>'])

        train_corp = vocab.words2ids_batch(train_corp)
        test_corp = vocab.words2ids_batch(test_corp)

        with StopWatch('Saving text (Main corpus)'):
            np.savetxt(cfg.processed_train_path, train_corp, fmt="%s")
            np.savetxt(cfg.processed_test_path, test_corp, fmt="%s")
            log.info("Saved preprocessed data: %s", cfg.processed_train_path)
        with StopWatch('Pickling vocab'):
            vocab.pickle(cfg.processed_vocab_path)
            log.info("Saved vocabulary: %s" % cfg.processed_vocab_path)
    else:
        log.info('Previously processed files will be used!')
        vocab = Vocab.unpickle(cfg.processed_vocab_path)
    StopWatch.stop('Total')
    return vocab


def process_pos_corpus(cfg):
    StopWatch.go('Total')

    if (not os.path.exists(cfg.pos_data_path)
        or not os.path.exists(cfg.pos_vocab_path)
        or cfg.reload_prepro):

        # load & process pos tags
        tag_proc = CorpusMultiProcessor(file_path=cfg.pos_path,
                                        min_len=cfg.min_len,
                                        max_len=cfg.max_len)
        tags, tag_counter = tag_proc.process()

        # make vocab
        tags_vocab = Vocab(counter=tag_counter,
                           embed_size=cfg.embed_size_t,
                           specials=['<eos>'])

        with StopWatch('Saving text (POS tagging corpus)'):
            np.savetxt(cfg.pos_data_path, tags_ids, fmt="%s")
            log.info("Saved preprocessed POS tags: %s", cfg.pos_data_path)

        with StopWatch('Pickling POS vocab'):
            tags_vocab.pickle(cfg.pos_vocab_path)
            log.info("Saved POS vocabulary: %s" % cfg.pos_vocab_path)
    else:
        log.info('Previously processed POS data files will be used!')
        tags_vocab = Vocab.unpickle(cfg.pos_vocab_path)

    StopWatch.stop('Total')
    return tags_vocab


def process_corpus_tag(cfg):
    StopWatch.go('Total')

    if (not os.path.exists(cfg.processed_train_path)
        or not os.path.exists(cfg.processed_vocab_path)
        or cfg.reload_prepro):

        log.info('Start preprocessing data and building vocabulary!')
        if isinstance(cfg.corpus_train_path, (list, tuple)):
            corpus_proc = CorpusTagMultiProcessor.from_multiple_files(
                file_paths=cfg.corpus_train_path,
                min_len=cfg.min_len,
                max_len=cfg.max_len)
            tokens, tags, token_cnt, tag_cnt = \
                CorpusTagMultiProcessor.multi_process(corpus_proc)
        else:
            corpus_proc = CorpusTagMultiProcessor(file_path=cfg.corpus_train_path,
                                                  min_len=cfg.min_len,
                                                  max_len=cfg.max_len)
            tokens, tags, token_cnt, tag_cnt = corpus_proc.process()

        # pretrained embedding initialization if necessary
        if cfg.load_glove:
            print('Loading GloVe pretrained embeddings...')
            glove_proc = GloveMultiProcessor(glove_dir=cfg.glove_dir,
                                             vector_size=cfg.embed_size_w)
            word2vec = glove_proc.process()
        else:
            word2vec = None

        # build sentence vocabulary & convert tokens to ids
        token_vocab = Vocab(counter=token_cnt,
                            embed_size=cfg.embed_size_w,
                            embed_init=word2vec,
                            max_size=cfg.vocab_size_w,
                            specials=['<pad>', '<sos>', '<eos>', '<unk>'])
        token_ids = token_vocab.words2ids_batch(tokens)
        cfg.vocab_size_w = len(token_vocab)

        # build tag vocabulary & convert tags to ids
        tag_vocab = Vocab(counter=tag_cnt,
                          embed_size=cfg.embed_size_t,
                          specials=['<pad>', '<sos>', '<eos>'])
        tags_ids = tag_vocab.words2ids_batch(tags)

        with StopWatch('Saving text (Main corpus)'):
            np.savetxt(cfg.processed_train_path, token_ids, fmt="%s")
            log.info("Saved preprocessed corpus: %s", cfg.processed_train_path)
            np.savetxt(cfg.pos_data_path, tags_ids, fmt="%s")
            log.info("Saved preprocessed POS tags: %s", cfg.pos_data_path)
        with StopWatch('Pickling vocab'):
            token_vocab.pickle(cfg.processed_vocab_path)
            log.info("Saved corpus vocabulary: %s" % cfg.processed_vocab_path)
            tag_vocab.pickle(cfg.pos_vocab_path)
            log.info("Saved POS tag vocabulary: %s" % cfg.pos_vocab_path)
    else:
        log.info('Previously processed files will be used!')
        token_vocab = Vocab.unpickle(cfg.processed_vocab_path)
        tag_vocab = Vocab.unpickle(cfg.pos_vocab_path)

    cfg.tag_size = len(tag_vocab)
    StopWatch.stop('Total')
    return token_vocab, tag_vocab


def process_pos_corpus_with_main_vocab(cfg, main_vocab):
    StopWatch.go('Total')

    if (not os.path.exists(cfg.pos_sent_data_path)
        or not os.path.exists(cfg.pos_tag_data_path)
        or not os.path.exists(cfg.pos_vocab_path)
        or cfg.reload_prepro):

        log.info('Start preprocessing data and building vocabulary!')
        sent_proc = CorpusMultiProcessor(file_path=cfg.pos_sent_path)
        sents, _ = sent_proc.process()
        sents_ids = main_vocab.words2ids_batch(sents)

        tag_proc = CorpusMultiProcessor(file_path=cfg.pos_tag_path)
        tags, tag_counter = tag_proc.process()
        tags_vocab = Vocab(counter=tag_counter,
                           embed_size=cfg.embed_size_t,
                           embed_init=word2vec,
                           specials=['<eos>'])
        tags_ids = tags_vocab.words2ids_batch(tags)

        with StopWatch('Saving text (POS tagging corpus)'):
            np.savetxt(cfg.pos_sent_data_path, sents_ids, fmt="%s")
            log.info("Saved preprocessed POS text: %s", cfg.pos_sent_data_path)
            np.savetxt(cfg.pos_tag_data_path, tags_ids, fmt="%s")
            log.info("Saved preprocessed POS tags: %s", cfg.pos_tag_data_path)

        with StopWatch('Pickling POS vocab'):
            tags_vocab.pickle(cfg.pos_vocab_path)
            log.info("Saved POS vocabulary: %s" % cfg.pos_vocab_path)
    else:
        log.info('Previously processed files will be used!')
        tags_vocab = Vocab.unpickle(cfg.pos_vocab_path)

    StopWatch.stop('Total')
    return tags_vocab
