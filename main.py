import logging
import os

from loader.corpus import CorpusDataset
from loader.process import process_main_corpus, process_pos_corpus
from test.test import test
from train.train import train
from train.network import Network
from utils.parser import parser
from utils.utils import Config, set_logger, prepare_paths


log = logging.getLogger('main')

if __name__ == '__main__':
    # Parsing arguments and set configs
    args = parser.parse_args()
    cfg = Config(vars(args))

    # Set all the paths
    prepare_paths(cfg)

    # Logger
    set_logger(cfg)
    log = logging.getLogger('main')

    # Preprocessing
    vocab_main = process_main_corpus(cfg)
    #vocab_pos= process_pos_corpus(cfg, vocab_books)

    # Load dataset
    corpus_main = CorpusDataset(cfg.corpus_data_path)
    #corpus_pos = CorpusDataset(cfg.pos_sent_data_path)

    # Build network
    net = Network(cfg, corpus_main, vocab_main)
    #net = Network(cfg, corpus_main, corpus_pos, vocab_main, vocab_pos)

    # Train
    if not cfg.test:
        train(net)
    # Test
    else:
        test(net)

    log.info('End of program.')
