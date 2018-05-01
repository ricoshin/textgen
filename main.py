import logging
import os

from loader.data import CorpusDataset, CorpusPOSDataset
from loader.process import process_main_corpus, process_corpus_tag
from test.test import Tester
from test.visualize import Visualizer
from train.train import Trainer
from train.network import Network
from utils.parser import parser
from utils.utils import Config, set_logger, prepare_paths


log = logging.getLogger('main')

if __name__ == '__main__':
    # Parsing arguments and set configs
    args = parser.parse_args()
    cfg = Config.init_from_parsed_args(args)

    # Set all the paths
    prepare_paths(cfg)

    # Logger
    set_logger(cfg)
    log = logging.getLogger('main')

    # Preprocessing & make dataset
    # if cfg.data_name = 'pos':
    #     vocab = process_main_corpus(cfg, 'split')
    #     vocab_pos = process_pos_corpus(cfg, 'split')
    #     corpus = CorpusPOSDataset(cfg.corpus_data_path,
    #                               cfg.pos_data_path)
    
    if cfg.pos_tag:
        vocab, vocab_tag = process_corpus_tag(cfg)
        corpus = CorpusPOSDataset(cfg.corpus_data_path,
                                  cfg.pos_data_path)
    else:
        vocab = process_main_corpus(cfg)
        vocab_tag = None
        corpus = CorpusDataset(cfg.corpus_data_path)

    # Build network
    net = Network(cfg, corpus, vocab, vocab_tag)

    # Train
    if not (cfg.test or cfg.visualize):
        Trainer(net)
    # Test
    else:
        if cfg.test:
            Tester(net)
        else:
            Visualizer(net)

    log.info('End of program.')
