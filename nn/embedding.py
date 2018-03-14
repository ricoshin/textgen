import logging
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.base_module import BaseModule
from utils.utils import to_gpu

log = logging.getLogger('main')


class WordEmbedding(BaseModule):
    def __init__(self, cfg, vocab):
        super(WordEmbedding, self).__init__()
        self.cfg = cfg
        self.vocab = vocab
        self.vocab_size = len(vocab)
        self.embed_size = cfg.word_embed_size
        #self.sos_batch = to_gpu(cfg.cuda, Variable(torch.ones(10,1).long()))
        self.embed = nn.Embedding(self.vocab_size, self.embed_size)

        # glove initialization
        if vocab.embed is not None:
            assert(vocab.embed.shape[0] == self.vocab_size)
            assert(vocab.embed.shape[1] == self.embed_size)
            self.embed.weight.data.copy_(torch.from_numpy(vocab.embed))
        else:
            self._init_weights()

        # fix embedding
        if cfg.fix_embed:
            self.requires_grad = False
        else:
            self.requires_grad = True

        self.embed.weight.requires_grad = self.requires_grad

    def _init_weights(self):
        # unifrom initialization in the range of [-0.1, 0.1]
        initrange = 0.1
        # Initialize Vocabulary Matrix Weight
        self.embedding.weight.data.uniform_(-initrange, initrange)

    def forward(self, indices, mode='hard'):
        assert(mode in ['hard', 'soft'])

        # normalize columns & zero PAD embedding
        new_weight = F.normalize(self.embed.weight, p=2, dim=1)
        new_weight[self.vocab.PAD_ID] = torch.zeros(self.embed_size)
        self.embed.weight = nn.Parameter(new_weight.data,
                                         requires_grad=self.requires_grad)

        if mode is 'hard':
            # indices : [bsz, max_len]
            assert(len(indices.size()) == 2)
            return self.embed(indices)
        else:
            # indices : [baz, max_len, vocab_size]
            assert(len(indices.size()) == 3)
            #assert(indices.size(1) == self.cfg.max_len)
            assert(indices.size(2) == self.vocab_size)
            return self.soft_embed(indices)

    def soft_embed(self, id_onehots):
        # id_onehot : [batch_size, max_len, vocab_size]
        # self.embed : [vocab_size, word_embed_size]
        max_len = id_onehots.size(1)
        vocab_size = id_onehots.size(2)
        word_embed_size = self.embed.weight.size(1)
        id_onehots = id_onehots.contiguous().view(-1, vocab_size)
        # id_onehots.size() : [bsz*max_len, vocab_size]
        embeddings = torch.mm(id_onehots, self.embed.weight)
        embeddings = embeddings.view(-1, max_len, word_embed_size)
        return embeddings

    def clip_grad_norm(self):
        nn.utils.clip_grad_norm(self.parameters(), self.cfg.clip)
        return self
