import numpy as np
import logging
import os

from loader.simple_questions import BookCorpusMultiProcessor
from loader.vocab import Vocab, GloveMultiProcessor
from utils.utils import StopWatch

import pdb

log = logging.getLogger('main')


def preprocess_data_vocab(cfg):
    StopWatch.go('Total')
    if (not os.path.exists(cfg.data_filepath)
        or not os.path.exists(cfg.vocab_filepath)
        or cfg.reload_prepro):

        log.info('Start preprocessing data and building vocabulary!')
        if isinstance(cfg.train_q_filepath, (list, tuple)):
            book_procs = BookCorpusMultiProcessor.from_multiple_files(
                    file_paths=cfg.train_q_filepath,
                    min_len=cfg.min_len,
                    max_len=cfg.max_len)
            sents, counter = BookCorpusMultiProcessor.multi_process(book_procs)
        else:
            book_procs = BookCorpusMultiProcessor(file_path=cfg.train_q_filepath,
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

        vocab = Vocab(counter=counter, max_size=cfg.vocab_size,
                      embed_dim=cfg.embed_size, init_embed=word2vec)

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

def preprocess_simpleqa(cfg):
    StopWatch.go('Total')
    if (not os.path.exists(cfg.train_data_filepath)
        or cfg.reload_prepro):

        log.info('Start preprocessing data and building vocabulary!')
        def get_idx_from_sents(filepath):
            if isinstance(filepath, (list, tuple)):
                procs = BookCorpusMultiProcessor.from_multiple_files(
                        file_paths=filepath, min_len=cfg.min_len, max_len=cfg.max_len)
                q_sents, a_sents, q_counter, a_counter = BookCorpusMultiProcessor.multi_process(procs)
            else:
                procs = BookCorpusMultiProcessor(file_path=filepath,
                                                      min_len=cfg.min_len,
                                                      max_len=cfg.max_len)
                q_sents, a_sents, q_counter, a_counter = procs.process()
            # pretrained embedding initialization if necessary
            if cfg.load_glove:
                print('Loading GloVe pretrained embeddings...')
                glove_processor = GloveMultiProcessor(glove_dir=cfg.glove_dir,
                                                      vector_size=cfg.embed_size)
                word2vec = glove_processor.process()
            else:
                word2vec = None

            q_vocab = Vocab(counter=q_counter, max_size=cfg.vocab_size,
                      embed_dim=cfg.embed_size, init_embed=word2vec)
            a_vocab = Vocab(counter=a_counter, max_size=cfg.vocab_size,
                      embed_dim=cfg.embed_size, init_embed=word2vec)
            return q_vocab.numericalize_sents(q_sents), a_vocab.numericalize_sents(a_sents), q_vocab, a_vocab

        q_sents, a_sents, q_vocab, a_vocab = get_idx_from_sents(cfg.train_filepath)
        sents = []
        # concatenate q_sent and a_sents
        for q, a in zip(q_sents, a_sents):
            pdb.set_trace()
            temp = q+'<spl>'+a
            sents.append()

        with StopWatch('Saving text'):
            np.savetxt(cfg.train_data_filepath, sents, fmt="%s")
            log.info("Saved preprocessed data: %s", cfg.train_data_filepath)
        with StopWatch('Pickling vocab'):
            q_vocab.pickle(cfg.q_vocab_filepath)
            a_vocab.pickle(cfg.a_vocab_filepath)
            log.info("Saved vocabulary: %s" % cfg.q_vocab_filepath)
            log.info("Saved vocabulary: %s" % cfg.a_vocab_filepath)
    else:
        log.info('Previously processed files will be used!')
        q_vocab = Vocab.unpickle(cfg.q_vocab_filepath)
        a_vocab = Vocab.unpickle(cfg.a_vocab_filepath)
    StopWatch.stop('Total')
    return q_vocab, a_vocab

def preprocess_simpleqa_separated(cfg):
    StopWatch.go('Total')
    if (not os.path.exists(cfg.train_q_data_filepath)
        or not os.path.exists(cfg.train_a_data_filepath)
        or cfg.reload_prepro):

        log.info('Start preprocessing data and building vocabulary!')
        def get_idx_from_sents(filepath):
            if isinstance(filepath, (list, tuple)):
                procs = BookCorpusMultiProcessor.from_multiple_files(
                        file_paths=filepath, min_len=cfg.min_len, max_len=cfg.max_len)
                q_sents, a_sents, q_counter, a_counter = BookCorpusMultiProcessor.multi_process(procs)
            else:
                procs = BookCorpusMultiProcessor(file_path=filepath,
                                                      min_len=cfg.min_len,
                                                      max_len=cfg.max_len)
                q_sents, a_sents, q_counter, a_counter = procs.process()
            # pretrained embedding initialization if necessary
            if cfg.load_glove:
                print('Loading GloVe pretrained embeddings...')
                glove_processor = GloveMultiProcessor(glove_dir=cfg.glove_dir,
                                                      vector_size=cfg.embed_size)
                word2vec = glove_processor.process()
            else:
                word2vec = None

            q_vocab = Vocab(counter=q_counter, max_size=cfg.vocab_size,
                      embed_dim=cfg.embed_size, init_embed=word2vec)
            a_vocab = Vocab(counter=a_counter, max_size=cfg.vocab_size,
                      embed_dim=cfg.embed_size, init_embed=word2vec)
            return q_vocab.numericalize_sents(q_sents), a_vocab.numericalize_sents(a_sents), q_vocab, a_vocab

        q_sents, a_sents, q_vocab, a_vocab = get_idx_from_sents(cfg.train_filepath)

        with StopWatch('Saving text'):
            np.savetxt(cfg.train_q_data_filepath, q_sents, fmt="%s")
            np.savetxt(cfg.train_a_data_filepath, a_sents, fmt="%s")
            log.info("Saved preprocessed data: %s", cfg.train_q_data_filepath)
            log.info("Saved preprocessed data: %s", cfg.train_a_data_filepath)
        with StopWatch('Pickling vocab'):
            q_vocab.pickle(cfg.q_vocab_filepath)
            a_vocab.pickle(cfg.a_vocab_filepath)
            log.info("Saved vocabulary: %s" % cfg.q_vocab_filepath)
            log.info("Saved vocabulary: %s" % cfg.a_vocab_filepath)
    else:
        log.info('Previously processed files will be used!')
        q_vocab = Vocab.unpickle(cfg.q_vocab_filepath)
        a_vocab = Vocab.unpickle(cfg.a_vocab_filepath)
    StopWatch.stop('Total')
    return q_vocab, a_vocab



# split simple questions dataset into question, answer files
import csv
def split_simple_questions(file_path):
    train_path = os.path.join(file_path, 'train_a.txt')
    test_path = os.path.join(file_path, 'test_a.txt')
    valid_path = os.path.join(file_path, 'vlid_a.txt')

    if (not os.path.exists(train_path)
        or not os.path.exists(test_path)
        or not os.path.exists(valid_path)
        or cfg.reload_prepro):
        log.info('splitting simple questions dataset')

        def write_qa_files(file_path, dataset_mode):
            file_name = dataset_mode + ".txt"
            f = open(os.path.join(file_path, file_name), 'r')
            reader = csv.reader(f, delimiter='\t')

            q_f = open(os.path.join(file_path, dataset_mode+"_q.txt"), 'w+')
            a_f = open(os.path.join(file_path, dataset_mode+"_a.txt"), 'w+')
            for row in reader:
                q_f.write(str(row[0])+'\n')
                a_f.write(str(row[1])+'\n')
            q_f.close()
            a_f.close()
            f.close()
        write_qa_files(file_path, 'train')
        write_qa_files(file_path, 'test')
        write_qa_files(file_path, 'valid')
