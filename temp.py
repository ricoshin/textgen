from collections import OrderedDict, namedtuple
import logging
import os

from tensorboardX import SummaryWriter
from tensorboardX.embedding import make_sprite, make_mat, append_pbtxt
import torch

log = logging.getLogger('main')


class ResultWriter(object):
    # result manager
    Embedding = namedtuple("Embedding", "embed, text")

    def __init__(self, cfg):
        self._outdir = cfg.name
        tf_events_path = os.path.join(cfg.log_dir, 'tf_events')
        self._writer = MySummaryWriter(tf_events_path)
        self.initialize()

    def initialize(self):
        """To initialize instance from outside"""
        self._scalar = OrderedDict()
        self._text = OrderedDict()
        self._embedding = OrderedDict()

    def __repr__(self):
        return self.__dict__.__repr__()

    def add(self, name, update_dict):
        """Update result dictionaries considering type of each update item.
        scalars, text : updated in a hierarchical dict structure
        embeddings : updated to only one-depth dict
            (as they have to be projected to the same space altogether.)

        """
        scalar = OrderedDict()
        text = OrderedDict()

        for sub_name, sub_value in update_dict.items():
            if type(sub_value) in (int, float):
                scalar.update({sub_name: sub_value})
            elif type(sub_value) is str:
                text.update({sub_name: sub_value})
            elif isinstance(sub_value, self.Embedding):
                self._embedding.update({name+'/'+sub_name: sub_value})
            else:
                raise Exception('Unknown type : %s' % type(sub_value))

        if scalar: self._scalar.update({name: scalar})
        if text: self._text.update({name: text})

    def log_scalar(self):
        if not self._scalar: return
        print_str = ""
        for name, scalar in self._scalar.items():
            print_str = "| %s | %s |" % (self._outdir, name)
            for sub_name, sub_scalar in scalar.items():
                print_str += " %s : %.8f |" % (sub_name, sub_scalar)
        log.info(print_str)

    def log_text(self):
        if not self._text: return
        print_str = ""
        for name, text in self._text.items():
            # print_str = "| %s | %s |\n" % (self._outdir, name)
            for sub_name, sub_text in text.items():
                print_str += sub_text
        log.info(print_str)

    def save_scalar(self, step):
        if not self._scalar: return
        for name, scalar in self._scalar.items():
            for sub_name, sub_scalar in scalar.items():
                tag = "%s/%s" % (name, sub_name)
                self._writer.add_scalar(tag, sub_scalar, step)

    def save_text(self, step):
        if not self._text: return

        for name, text in self._text.items():
            for sub_name, sub_text in text.items():
                tag = "%s/%s" % (name, sub_name)
                self._writer.add_text(tag, sub_text, step)

    def save_embedding(self, step):
        """Append all embeddings with the same tag but different metadata"""
        if not self._embedding: return
        all_embed = []
        all_label = []
        all_id = []
        all_text = []
        for name, (embed, text) in self._embedding.items():
            if type(embed) is torch.cuda.LongTensor:
                embed = embed.cpu()  # better do this only when necessary
            all_embed.append(embed)
            all_label += ([name] * embed.size(0))
            all_id += ["[%s]" % str(i) for i in range(embed.size(0))]
            all_text += text
        all_embed = torch.cat(all_embed, dim=0)
        metadata = dict(label=all_label, id=all_id, text=all_text)
        self._writer.add_embedding(all_embed, metadata,
                                   global_step=step, tag='embedding')


class MySummaryWriter(SummaryWriter):
    def add_embedding(self, mat, metadata=None, label_img=None,
                      global_step=None, tag='default'):
        """Override make_tsv function in order to handle multi-column tsv file.

        Args(changed):
            metadata (dict): Keys -> headers of metadata
                             Values -> values of metatdata
        """
        if global_step is None:
            global_step = 0

        save_path = os.path.join(self.file_writer.get_logdir(),
                                 str(global_step).zfill(5))
        try:
            os.makedirs(save_path)
        except OSError:
            # to control log level
            info.warning('warning: Embedding dir exists, '
                         'did you set global_step for add_embedding()?')

        if metadata is not None:
            assert all(mat.size(0) == len(d) for d in metadata.values()), \
                   '#labels should equal with #data points'
            make_tsv(metadata, save_path)

        if label_img is not None:
            assert mat.size(0) == label_img.size(0), \
                  '#images should equal with #data points'
            make_sprite(label_img, save_path)

        assert mat.dim() == 2, \
               'mat should be 2D and mat.size(0) is the number of data points'
        make_mat(mat.tolist(), save_path)
        # new funcion to append to the config file a new embedding
        append_pbtxt(metadata, label_img, self.file_writer.get_logdir(),
                     str(global_step).zfill(5), tag)


def make_tsv(metadata, save_path):
    """Only used in MySummaryWriter. Write multi-column tsv metadata file"""
    metadata_str = []
    process = lambda data : '\t'.join([str(d) for d in data])
    metadata_str.append(process(metadata.keys()))

    for data in zip(*[value for value in metadata.values()]):
        metadata_str.append(process(data))

    with open(os.path.join(save_path, 'metadata.tsv'), 'w') as f:
        for x in metadata_str:
            f.write(x + '\n')
