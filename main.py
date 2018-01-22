import logging
import os

from dataloader.book_corpus import BookCorpusDataset
from dataloader.preprocess import preprocess_data_vocab
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
    vocab = preprocess_data_vocab(cfg)

    # Load dataset
    book_corpus = BookCorpusDataset(cfg.data_filepath)

    # Build network
    net = Network(cfg, book_corpus, vocab)

    # Train
    if not cfg.test:
        train(net)
    # Test
    else:
        test(net)

    log.info('End of program.')
