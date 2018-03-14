from collections import OrderedDict, namedtuple
import logging
import os

from tensorboardX import SummaryWriter
from tensorboardX.embedding import make_sprite, make_mat, append_pbtxt
import torch

log = logging.getLogger('main')


class ResultWriter(object):
    Embedding = namedtuple("Embedding", "embed, text")

    def __init__(self, cfg):
        filename = os.path.join(cfg.log_dir, 'tf_events')
        self._writer = MySummaryWriter(filename)
        self.initialize_scalar_text()
        self.initialize_embedding()

    def __repr__(self):
        return self.__dict__.__repr__()

    def initialize_scalar_text(self):
        self._scalar_text = OrderedDict()

    def initialize_embedding(self):
        self._embedding = OrderedDict()

    def add(self, label, dict_):
        scalar_text_pack = ScalarTextPack()
        for name, value in dict_.items():
            if type(value) in (int, float, str):
                scalar_text_pack.add({name: value})
            elif type(value) is self.Embedding:
                name = label + '/' + name
                self._embedding.update({name: value})
            else:
                raise Exception('Unknown type : %s' % type(value))
        self._scalar_text.update({label: scalar_text_pack})

    def _str_scalar_in_pack(self, pack):
        outstr = ""
        for name, scalar in pack.named_scalar():
            outstr += " %s : %.8f |" % (name, scalar)
        return outstr

    def _str_text_in_pack(self, pack):
        outstr = ""
        for name, text in pack.named_text():
            outstr += text
        return outstr

    def log_scalar(self):
        for label, pack in self._scalar_text.items():
            header = "| %s |" % label
            log.info(header + self._str_scalar_in_pack(pack))

    def log_text(self):
        for label, pack in self._scalar_text.items():
            header = "| %s |" % label
            log.info(header + self._str_text_in_pack(pack))

    def log_scalar_text(self):
        for label, pack in self._scalar_text.items():
            header = "| %s |" % label
            log.info(header + self._str_scalar_in_pack(pack))
            log.info(self._str_text_in_pack(pack))

    def save_scalar(self, step):
        for label, pack in self._scalar_text.items():
            for name, scalar in pack.named_scalar():
                tag = "%s/%s" % (label, name)
                self._writer.add_scalar(tag, scalar, step)

    def save_text(self, step):
        for label, pack in self._scalar_text.items():
            for name, text in pack.named_text():
                tag = "%s/%s" % (label, name)
                self._writer.add_text(tag, text, step)

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


class ScalarTextPack(object):
    def __init__(self):
        self._scalar = OrderedDict()
        self._text = OrderedDict()

    def __len__(self):
        return len(self._scalar) + len(self._text)

    def __bool__(self):
        return len(self) > 0

    def add(self, dict_):
        for key, value in dict_.items():
            if type(value) in (int, float):
                self._scalar.update({key: value})
            elif type(value) is str:
                self._text.update({key: value})
            else:
                raise Exception('Unknown type : %s' % type(value))

    def named_scalar(self):
        for name, scalar in self._scalar.items():
            yield name, scalar

    def named_text(self):
        for name, text in self._text.items():
            yield name, text

    def named_all(self):
        out_dict = dict()
        out_dict.update(self._scalar).update(self._text)
        for name, value in out_dict.items():
            yield name, value


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
                                 str(global_step).zfill(9))
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
                     str(global_step).zfill(9), tag)


def make_tsv(metadata, save_path):
    """Used only in MySummaryWriter. Write multi-column tsv metadata file"""
    metadata_str = []
    process = lambda data : '\t'.join([str(d) for d in data])
    metadata_str.append(process(metadata.keys()))

    for data in zip(*[value for value in metadata.values()]):
        metadata_str.append(process(data))

    with open(os.path.join(save_path, 'metadata.tsv'), 'w') as f:
        for x in metadata_str:
            f.write(x + '\n')
