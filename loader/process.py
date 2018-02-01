import numpy as np
import logging
import os

from loader.corpus import CorpusMultiProcessor
from loader.vocab import Vocab, GloveMultiProcessor
from utils.utils import StopWatch

log = logging.getLogger('main')


def process_main_corpus(cfg):
    StopWatch.go('Total')
    if (not os.path.exists(cfg.corpus_data_path)
        or not os.path.exists(cfg.corpus_vocab_path)
        or cfg.reload_prepro):

        log.info('Start preprocessing data and building vocabulary!')
        if isinstance(cfg.corpus_path, (list, tuple)):
            corpus_proc = CorpusMultiProcessor.from_multiple_files(
                file_paths=cfg.corpus_path,
                min_len=cfg.min_len,
                max_len=cfg.max_len)
            sents, counter = CorpusMultiProcessor.multi_process(corpus_proc)
        else:
            corpus_proc = CorpusMultiProcessor(file_path=cfg.corpus_path,
                                               min_len=cfg.min_len,
                                               max_len=cfg.max_len)
            sents, counter = corpus_proc.process()

        # pretrained embedding initialization if necessary
        if cfg.load_glove:
            print('Loading GloVe pretrained embeddings...')
            glove_proc = GloveMultiProcessor(glove_dir=cfg.glove_dir,
                                            vector_size=cfg.embed_size)
            word2vec = glove_proc.process()
        else:
            word2vec = None

        vocab = Vocab(counter=counter, max_size=cfg.vocab_size)
        vocab.generate_embeddings(embed_dim=cfg.embed_size, init_embed=word2vec)
        sents = vocab.numericalize_sents(sents)


        with StopWatch('Saving text (Main corpus)'):
            np.savetxt(cfg.corpus_data_path, sents, fmt="%s")
            log.info("Saved preprocessed data: %s", cfg.corpus_data_path)
        with StopWatch('Pickling vocab'):
            vocab.pickle(cfg.corpus_vocab_path)
            log.info("Saved vocabulary: %s" % cfg.corpus_vocab_path)
    else:
        log.info('Previously processed files will be used!')
        vocab = Vocab.unpickle(cfg.corpus_vocab_path)
    StopWatch.stop('Total')
    return vocab


def process_pos_corpus(cfg, main_vocab):
    StopWatch.go('Total')

    if (not os.path.exists(cfg.pos_sent_data_path)
        or not os.path.exists(cfg.pos_tag_data_path)
        or not os.path.exists(cfg.pos_vocab_path)
        or cfg.reload_prepro):

        log.info('Start preprocessing data and building vocabulary!')
        sent_proc = CorpusMultiProcessor(file_path=cfg.pos_sent_path)
        sents, _ = sent_proc.process()
        sents_ids = main_vocab.numericalize_sents(sents)

        tag_proc = CorpusMultiProcessor(file_path=cfg.pos_tag_path)
        tags, tag_counter = tag_proc.process()
        tags_vocab = Vocab(counter=tag_counter, specials=False)
        tags_ids = tags_vocab.numericalize_sents(tags)

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
