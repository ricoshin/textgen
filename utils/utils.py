import logging
import numpy as np
import os
import random
from time import time, strftime, gmtime

import torch

from models.cnn_architect import ConvnetType, ConvnetArchitect

log = logging.getLogger('main')


class StopWatch(object):
    time = dict()
    history = dict()

    def __init__(self, name):
        self.name = name
        StopWatch.go(name)

    def __enter__(self):
        pass

    def __exit__(self, type, value, trace_back):
        StopWatch.stop(self.name)
        del self.name

    @classmethod
    def go(cls, name):
        cls.time[name] = time()

    @classmethod
    def stop(cls, name, print=True):
        start_time = cls.time.get(name, None)
        if start_time:
            elapsed_time = time() - start_time
            cls.print_elapsed_time(name, elapsed_time)
            cls.history[name] = elapsed_time
            del cls.time[name]
        else:
            log.info('Not registered name : %s' % name)

    @classmethod
    def print_elapsed_time(cls, name, seconds):
        msg = "StopWatch [%s] : %5f " % (name, seconds)
        hms = strftime("(%Hhrs %Mmins %Ssecs)", gmtime(seconds))
        log.info(msg + hms)


class Config(object):
    def __init__(self, init=None):
        if init is None:
            return
        self.update(init)

    @classmethod
    def init_from_parsed_args(cls, args):
        cfg = cls(vars(args))
        architect = ConvnetArchitect(cfg)
        arch = architect.design_model_of(ConvnetType.AUTOENCODER)
        cfg.update(dict(arch_cnn=Config(arch)))
        return cfg

    def update(self, new_config):
        self.__dict__.update(new_config)

    def __repr__(self):
        return self.__dict__.__repr__()


class MyFormatter(logging.Formatter):
    info_fmt = "%(message)s"
    else_fmt = '[%(levelname)s] %(message)s'

    def __init__(self, fmt="%(message)s"):
        logging.Formatter.__init__(self, fmt)

    def format(self, record):
        # Save the original format configured by the user
        # when the logger formatter was instantiated
        format_orig = self._fmt
        # Replace the original format with one customized by logging level
        if record.levelno == logging.INFO:
            self._fmt = MyFormatter.info_fmt
        else:
            self._fmt = MyFormatter.else_fmt
        # Call the original formatter class to do the grunt work
        result = logging.Formatter.format(self, record)
        # Restore the original format configured by the user
        self._fmt = format_orig

        return result


def set_logger(cfg):
    #log_fmt = '%(asctime)s %(levelname)s %(message)s'
    #date_fmt = '%d/%m/%Y %H:%M:%S'
    #formatter = logging.Formatter(log_fmt, datefmt=date_fmt)
    formatter = MyFormatter()

    # set log level
    levels = dict(debug=logging.DEBUG,
                  info=logging.INFO,
                  warning=logging.WARNING,
                  error=logging.ERROR,
                  critical=logging.CRITICAL)

    log_level = levels.get(cfg.log_level)


    # setup file handler
    file_handler = logging.FileHandler(cfg.log_path)
    file_handler.setFormatter(formatter)
    file_handler.setLevel(log_level)

    # setup stdio handler
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    stream_handler.setLevel(log_level)

    # get logger
    logger = logging.getLogger('main')
    logger.setLevel(log_level)

    # add file & stdio handler to logger
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)


def prepare_paths(cfg):
    sanity_check(cfg)
    # set directories
    cfg.log_dir = os.path.join(cfg.out_dir, cfg.name)
    cfg.data_dir = os.path.join(cfg.data_dir, cfg.data_name)
    prepro_name = "%s_%d_%d" % (cfg.data_name, cfg.min_len, cfg.max_len)
    cfg.prepro_dir = os.path.join(cfg.prepro_dir, prepro_name)
    cfg.log_path = os.path.join(cfg.log_dir, "log.txt")

    # pos corpus& tags filepath
    cfg.pos_sent_path = './data/pos_tagging/train/sentences.txt'
    cfg.pos_tag_path = './data/pos_tagging/train/tags.txt'

    # main corpus filepath
    if cfg.small:
        cfg.word_embed_size = 50

    if cfg.data_name == "books":
        if cfg.small:
            cfg.prepro_dir += "_small"
            filename = "books_1k.txt"
            cfg.corpus_path = os.path.join(cfg.data_dir, filename)
        else:
            filenames = ["books_large_p1.txt", "books_large_p2.txt"]
            cfg.corpus_path = \
                [*map(lambda fn: os.path.join(cfg.data_dir, fn), filenames)]

    elif cfg.data_name == "snli":
        if cfg.small:
            raise Exception("There's no small version of snli dataset!")
        else:
            cfg.corpus_path = os.path.join(cfg.data_dir, 'train.txt')

    elif cfg.data_name == "pos":
        if cfg.small:
            raise Exception("There's no small version of pos dataset!")
        else:
            cfg.corpus_path = os.path.join(cfg.data_dir, 'train/sentences.txt')
            cfg.pos_path = os.path.join(cfg.data_dir, 'train/tags.txt')

    # preprocessed file path
    cfg.corpus_data_path = os.path.join(cfg.prepro_dir, "data.txt")
    cfg.corpus_vocab_path = os.path.join(cfg.prepro_dir, "vocab.pickle")

    if cfg.pos_tag:
        cfg.pos_data_path = os.path.join(cfg.prepro_dir, "data_pos.txt")
        cfg.pos_vocab_path = os.path.join(cfg.prepro_dir, "vocab_pos.pickle")

    # make dirs if not exists
    if not os.path.exists(cfg.data_dir):
        raise Exception("can't find data_dir: %s" % cfg.data_dir)

    if not os.path.exists(cfg.glove_dir) and load_glove:
        raise Exception("cant't find glove_dir: %s" % cfg.glove_dir)

    if not os.path.exists(cfg.log_dir): # this includes out_dir
        os.makedirs(cfg.log_dir)

    if not os.path.exists(cfg.prepro_dir):
        os.makedirs(cfg.prepro_dir)

def sanity_check(cfg, strict=False):

    def inform_fn(msg):
        if strict:
            raise Exception(msg)
        else:
            log.warning(msg)

    if cfg.fix_embed and not cfg.load_glove:
        inform_fn('cfg.load_glove is False, but trying fix_embed.')

def set_random_seed(cfg):
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    if cfg.cuda and torch.cuda.is_available():
        torch.cuda.manual_seed(cfg.seed)


def to_gpu(gpu, var):
    if gpu:
        return var.cuda()
    return var
