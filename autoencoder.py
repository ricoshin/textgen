import logging

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


from utils import to_gpu

log = logging.getLogger('main')

class Autoencoder(nn.Module):
    def __init__(self, emsize, nhidden, ntokens, nlayers, noise_radius=0.2,
                 hidden_init=False, dropout=0, gpu=False):
        super(Autoencoder, self).__init__()
        self.nhidden = nhidden
        self.emsize = emsize
        self.ntokens = ntokens
        self.nlayers = nlayers
        self.noise_radius = noise_radius
        self.hidden_init = hidden_init
        self.dropout = dropout
        self.gpu = gpu

        self.start_symbols = to_gpu(gpu, Variable(torch.ones(10, 1).long()))

        # Vocabulary embedding
        self.embedding = nn.Embedding(ntokens, emsize)
        self.embedding_decoder = nn.Embedding(ntokens, emsize)

        # RNN Encoder and Decoder
        self.encoder = nn.LSTM(input_size=emsize,
                               hidden_size=nhidden,
                               num_layers=nlayers,
                               dropout=dropout,
                               batch_first=True)

        decoder_input_size = emsize+nhidden
        self.decoder = nn.LSTM(input_size=decoder_input_size,
                               hidden_size=nhidden,
                               num_layers=1,
                               dropout=dropout,
                               batch_first=True)

        # Initialize Linear Transformation
        self.linear = nn.Linear(nhidden, ntokens)

        self.criterion_ce = nn.CrossEntropyLoss()

        self.init_weights()


    def init_weights(self):
        # unifrom initialization in the range of [-0.1, 0.1]
        initrange = 0.1

        # Initialize Vocabulary Matrix Weight
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.embedding_decoder.weight.data.uniform_(-initrange, initrange)
        # by default it's initialized with normal_(0,1)

        # Initialize Encoder and Decoder Weights
        for p in self.encoder.parameters():
            p.data.uniform_(-initrange, initrange)
        for p in self.decoder.parameters():
            p.data.uniform_(-initrange, initrange)

        # Initialize Linear Weight
        self.linear.weight.data.uniform_(-initrange, initrange)
        self.linear.bias.data.fill_(0)

    def init_hidden(self, bsz):
        zeros1 = Variable(torch.zeros(self.nlayers, bsz, self.nhidden))
        zeros2 = Variable(torch.zeros(self.nlayers, bsz, self.nhidden))
        return (to_gpu(self.gpu, zeros1), to_gpu(self.gpu, zeros2))

    def init_state(self, bsz):
        zeros = Variable(torch.zeros(self.nlayers, bsz, self.nhidden))
        return to_gpu(self.gpu, zeros)

    def store_enc_grad_norm(self, grad):
        norm = torch.norm(grad, 2, 1)
        self.enc_grad_norm = norm.detach().data.mean()
        # use this when compute disc_c's gradient (register_hook)
        return grad

    def store_dec_grad_norm(self, grad):
        norm = torch.norm(grad, 2, 1)
        self.dec_grad_norm = norm.detach().data.mean()
        # use this when compute disc_s's gradient (register_hook)
        return grad

    def forward(self, indices, lengths, noise, encode_only=False):
        batch_size, maxlen = indices.size()

        hidden = self.encode(indices, lengths, noise)

        if encode_only:
            return hidden # batch_size x hidden_size

        if hidden.requires_grad:
            hidden.register_hook(self.store_enc_grad_norm)
            #log.debug("Encoder gradient norm has been saved.")

        decoded = self.decode(hidden, batch_size, maxlen,
                              indices=indices, lengths=lengths)

        return decoded

    def encode(self, indices, lengths, noise):
        # indices.size() : batch_size x max(lengths) [Variable]
        # len(lengths) : batch_size [List]
        embeddings = self.embedding(indices)
        # embeddings.data.size() : batch_size x max(lenghts) x embed_dim [Variable]
        packed_embeddings = pack_padded_sequence(input=embeddings,
                                                 lengths=lengths,
                                                 batch_first=True)
        # packed_embeddings.data.size() : sum(lengths) x embed_dim [Variable]
        # len(packed_embeddings.batch_sizes) : max(lengths) [list]

        # Encode
        packed_output, state = self.encoder(packed_embeddings)
        # packed_output.data.size() : sum(length) x hidden_size [Variable]
        # len(packed_output.batch_sizes) : max(length) [list]

        # if you do :
        #   result = pad_packed_sequence(packed_outputs, batch_first=True)
        #   then, type(result) == tuple, len(result) == 2
        #   result[0].size() : batch_size x max(lengths) x hidden_size [Variable]
        #   len(result[1]) : bath_size [List]

        hidden, cell = state # last states (tuple the length of 2)
        # (hidden states & cell states)
        # hidden.size() : 1 x batch_size x hidden_size
        # cell.size() : 1 x batch_size x hidden_size
        # 1 : (num_directions) * (num_layers)

        # result[0][i][lengths[i]-1] == hidden[0][i]
        #   state only captures the (actual) last state of each batch

        # batch_size x hidden_size
        hidden = hidden[-1]  # get hidden state of last layer of encoder

        # normalize to unit ball (l2 norm of 1) - p=2, dim=1
        norms = torch.norm(hidden, 2, 1)

        # For older versions of PyTorch use:
        # hidden = torch.div(hidden, norms.expand_as(hidden))
        # For newest version of PyTorch (as of 8/25) use this:
        hidden = torch.div(hidden, norms.unsqueeze(1).expand_as(hidden))

        if noise and self.noise_radius > 0:
            gauss_noise = torch.normal(means=torch.zeros(hidden.size()),
                                       std=self.noise_radius)
            hidden = hidden + to_gpu(self.gpu, Variable(gauss_noise))

        return hidden # batch_size x hidden_size

    def decode(self, hidden, batch_size, maxlen, indices=None, lengths=None):
        # batch x hidden
        # hidden.size() : batch_size x hidden_size
        # hidden.unsqueeze(1) : batch_size x 1 x hidden_size
        # hidden.unsqueeze(1).repeat(1, maxlen, 1) : batch_size x max_len x hidden_size
        all_hidden = hidden.unsqueeze(1).repeat(1, maxlen, 1)
        # duplicate hidden states along all time steps (to be augumented to inputs)

        if self.hidden_init:
            # initialize decoder hidden state to encoder output
            # (plus) initialze decoder cell state to zero state
            state = (hidden.unsqueeze(0), self.init_state(batch_size))
            # self.init_state(bsz) :
            #   Variable(torch.zeros(self.nlayers, bsz, self.nhidden))
        else:
            # initialize both states to zero states
            state = self.init_hidden(batch_size)

        embeddings = self.embedding_decoder(indices) # for teacher forcing
        augmented_embeddings = torch.cat([embeddings, all_hidden], 2)
        packed_embeddings = pack_padded_sequence(input=augmented_embeddings,
                                                 lengths=lengths,
                                                 batch_first=True)

        packed_output, state = self.decoder(packed_embeddings, state)
        output, lengths = pad_packed_sequence(packed_output, batch_first=True)
        # output.size() : batch_size x max(lengths) x hidden_size
        # len(lengths) : batch_size

        if output.requires_grad:
            output.register_hook(self.store_dec_grad_norm)
            #log.debug("Decoder gradient norm has been saved.")

        # reshape to batch_size*maxlen x nhidden before linear over vocab
        decoded = self.linear(output.contiguous().view(-1, self.nhidden))
        decoded = decoded.view(batch_size, maxlen, self.ntokens)

        return decoded # decoded.size() : batch x max_len x ntokens [logits]

    def mask_output_target(self, output, target, ntokens):
        # Create sentence length mask over padding
        target_mask = target.gt(0) # greater than 0
        # [1,1,..,1,1,1,1,1,0,
        #  1,1,..,1,1,1,1,0,0,
        #          ...
        #  1,1,..,1,1,1,0,0,0]
        masked_target = target.masked_select(target_mask)
        # target_mask.size(0) = batch_size*max_len
        # output_mask.size() : batch_size*max_len x ntokens
        target_mask = target_mask.unsqueeze(1)
        output_mask = target_mask.expand(target_mask.size(0), ntokens)
        # torch.Tensor.expand(*sizes)
        #  : returns a new view of the tensor with singleton dimensions
        #    expanded to a larger size.

        # flattened_output.size(): batch_size*max_len x ntokens
        flattened_output = output.view(-1, ntokens)

        # flattened_output.masked_select(output_mask).size()
        #  num_of_masked_words(in batch, excluding <pad>)*ntokens
        masked_output = flattened_output.masked_select(output_mask)
        masked_output = masked_output.view(-1, ntokens)
        # masked_output.size() : num_of_masked_words x ntokens
        # masked_target : num_of_masked_words

        return masked_output, masked_target

    @staticmethod
    def train_(cfg, ae, batch):
        ae.train()
        ae.zero_grad()

        source, target, lengths = batch
        source = to_gpu(cfg.cuda, Variable(source))
        target = to_gpu(cfg.cuda, Variable(target)) # requires_grad=False

        # output.size(): batch_size x max_len x ntokens (logits)
        output = ae(source, lengths, noise=True)

        masked_output, masked_target = \
            ae.mask_output_target(output, target, ae.ntokens)

        max_vals, max_indices = torch.max(masked_output, 1)
        accuracy = torch.mean(max_indices.eq(masked_target).float())

        loss = ae.criterion_ce(masked_output/cfg.temp, masked_target)
        # criterion_ce == torch.nn.CrossEntropyLoss
        # criterion_ce(input, class)
        # input : scores for each class / size = (batch_size, class_num)
        # class : class index (0 to class_num-1) / size = (batch_size)

        loss.backward()
        # `clip_grad_norm` to prevent exploding gradient in RNNs / LSTMs
        torch.nn.utils.clip_grad_norm(ae.parameters(), cfg.clip) # cfg.clip : 1(default)
        # clip_grad_norm(parameters, max_norm, norm_type=2)

        return loss.data, accuracy.data[0]

    @staticmethod
    def eval_(cfg, ae, batch):
        ae.eval()

        source, target, lengths = batch
        source = to_gpu(cfg.cuda, Variable(source, volatile=True))
        target = to_gpu(cfg.cuda, Variable(target, volatile=True)) # requires_grad=False

        # output.size(): batch_size x max_len x ntokens (logits)
        output = ae(source, lengths, noise=True)

        masked_output, masked_target = \
            ae.mask_output_target(output, target, ae.ntokens)

        max_value, max_indices = torch.max(output, 2)
        target = target.view(output.size(0), -1)
        outputs = max_indices.data.cpu().numpy()
        targets = target.data.cpu().numpy()

        return targets, outputs

    @staticmethod
    def encode_(cfg, ae, batch, train=True):
        if train:
            ae.train()
            ae.zero_grad()
        else:
            ae.eval()

        source, target, lengths = batch
        source = to_gpu(cfg.cuda, Variable(source))
        target = to_gpu(cfg.cuda, Variable(target))

        # when encode_only is True:
        #   encoded hidden states (batch_size x hidden_size) are returned
        real_hidden = ae(source, lengths, noise=False, encode_only=True)
        return real_hidden

    @staticmethod
    def decode_(cfg, ae, hidden, train=True):
        if train:
            ae.train()
            ae.zero_grad()
        else:
            ae.eval()

        if not (train and cfg.backprop_gen):
            # should not backprop gradient to enc/gen
            hidden = hidden.detach()

        batch_size = hidden.size(0)

        if ae.hidden_init:
            # initialize decoder hidden state to encoder output
            state = (hidden.unsqueeze(0), ae.init_state(batch_size))
        else:
            state = ae.init_hidden(batch_size)

        # <sos>
        ae.start_symbols.data.resize_(batch_size, 1)
        # Tensor.resize_(): resize tensor to the specified sizes(could be larger or smaller)
        ae.start_symbols.data.fill_(1)
        # self.start_symbols.size() : [batch_size, 1]

        embedding = ae.embedding_decoder(ae.start_symbols)
        # embedding : [batch_size, 1, embedding_size]
        inputs = torch.cat([embedding, hidden.unsqueeze(1)], 2)

        # unroll
        all_indices = []
        all_hiddens = []
        max_len = cfg.max_len + 1 # including sos/eos
        for i in range(max_len): # for each step
            # input very first step (sos totken + fake_z_code)
            output, state = ae.decoder(inputs, state)
            # output.size() : bath_size x 1(max_len) x nhidden
            # state.size() : 1(num_layer) x batch_size x nhidden
            overvocab = ae.linear(output.squeeze(1))
            # output.squeeze(1) : batch_size x nhidden (output layer)
            if not cfg.sample:
                vals, indices = torch.max(overvocab, 1)
            else: # sampling
                probs = F.softmax(overvocab/temp) # get probabilities that sum to 1
                indices = torch.multinomial(probs, 1).squeeze(1) # multinomial sampling

            all_hiddens.append(output)
            all_indices.append(indices.unsqueeze_(1)) # append generated vocab at each step
            embedding = ae.embedding_decoder(indices)
            inputs = torch.cat([embedding, hidden.unsqueeze(1)], 2)

        max_indices = torch.cat(all_indices, 1)
        hidden_states = torch.cat(all_hiddens, 1)

        # max_indices.size() : batch_size x max_len
        # hidden_states.size() : batch_size x max_len X nhidden
        max_indices = max_indices.data.cpu().numpy()
        return max_indices, hidden_states

    @staticmethod
    def decoder_train_(cfg, ae, disc_s, fake_code):
        ae.train()
        ae.zero_grad()

        fake_ids, fake_states = ae.decode_(cfg, ae, fake_code)

        def grad_hook(grad):
            if cfg.ae_grad_norm:
                gan_norm = torch.norm(grad, 2, 1).detach().data.mean()
                if gan_norm == .0:
                    log.warning("zero sample_gan norm!")
                    normed_grad = grad
                else:
                    normed_grad = grad * ae.enc_grad_norm / gan_norm
            else:
                normed_grad = grad
            return normed_grad

        fake_states.register_hook(grad_hook)

        err_dec, attn_fake = disc_s(fake_states)

        # loss / backprop
        one = to_gpu(cfg.cuda, torch.FloatTensor([1]))
        err_dec.backward(one)

        err_dec = err_dec.data[0]
        return err_dec
