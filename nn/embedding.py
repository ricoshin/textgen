import logging
import torch
import torch.nn as nn

from utils.utils import to_gpu

log = logging.getLogger('main')


class WordEmbedding(nn.Module):
    def __init__(self, vocab_size, embed_size, fix_embed, init_embed=None):
        super(WordEmbedding, self).__init__()
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        #self.sos_batch = to_gpu(cfg.cuda, Variable(torch.ones(10,1).long()))
        self.embed = nn.Embedding(vocab_size, embed_size)

        # glove initialization
        if init_embed is not None:
            assert(init_embed.shape[0] == vocab_size)
            assert(init_embed.shape[1] == embed_size)
            self.embed.weight.data.copy_(torch.from_numpy(init_embed))
        else:
            self._init_weights()

        # fix embedding
        if fix_embed:
            self.embed.weight.requires_grad = False


    def _init_weights(self):
        # unifrom initialization in the range of [-0.1, 0.1]
        initrange = 0.1
        # Initialize Vocabulary Matrix Weight
        self.embedding.weight.data.uniform_(-initrange, initrange)


    def forward(self, indices, mode='hard'):
        assert(mode in ['hard', 'soft'])
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
