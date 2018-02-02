import logging
import os

from loader.corpus import CorpusDataset, CorpusPOSDataset
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

    # Preprocessing & make dataset
    if cfg.data_name == 'pos':
        vocab = process_main_corpus(cfg, 'split')
        vocab_pos = process_pos_corpus(cfg, 'split')
        corpus = CorpusPOSDataset(cfg.corpus_data_path,
                                  cfg.pos_data_path)
    else:
        vocab = process_main_corpus(cfg, 'spacy')
        vocab_pos = None
        corpus = CorpusDataset(cfg.corpus_data_path)

    # Build network
    net = Network(cfg, corpus, vocab, vocab_pos)

    # Train
    if not cfg.test:
        train(net)
    # Test
    else:
        test(net)

    log.info('End of program.')
