from models.encoder import EncoderCNN, EncoderRNN
from models.decoder import DecoderCNN, DecoderRNN
from models.base_module import BaseModule

class BaseAutoencoder(BaseModule):
    def __init__(self, embedding, encoder, decoder):
        super(AutoencoderCNN, self).__init__()
        self.embed = embedding
        self.enc = encoder
        self.dec = decoder

        # placeholder
        self.code = None
        self.embed = None

    def forward(self, batch):
        return self._autoencode(batch)

    def code_hook(self, hook):
        self.code.register_hook(hook)

    def embed_hook(self, hook):
        self.embed.register_hook(hook)

    def clip_grad_norm_(self):
        nn.utils.clip_grad_norm_(self.parameters(), self.cfg.clip)
        return self

    def _loss_and_accuracy(self, prob, target, vocab_size):
        masked_output, masked_target = \
            self._mask_output_target(prob, target, vocab_size)
        loss = self.net.dec.criterion_nll(masked_output, masked_target)
        _, max_ids = torch.max(masked_output, 1)
        acc = torch.mean(max_ids.eq(masked_target).float())

        return loss, acc

    def _mask_output_target(output, target, ntokens):
        # Create sentence length mask over padding
        target_mask = target.gt(0) # greater than 0
        masked_target = target.masked_select(target_mask)
        # target_mask.size(0) = batch_size*max_len
        # output_mask.size() : batch_size*max_len x ntokens
        target_mask = target_mask.unsqueeze(1)
        output_mask = target_mask.expand(target_mask.size(0), ntokens)
        # flattened_output.size(): batch_size*max_len x ntokens
        flattened_output = output.view(-1, ntokens)
        # flattened_output.masked_select(output_mask).size()
        #  num_of_masked_words(in batch, excluding <pad>)*ntokens
        masked_output = flattened_output.masked_select(output_mask)
        masked_output = masked_output.view(-1, ntokens)
        # masked_output.size() : num_of_masked_words x ntokens
        # masked_target : num_of_masked_words
        return masked_output, masked_target


class AutoencoderRNN(BaseAutoencoder):
    def __init__(self, *args):
        super(AutoencoderRNN, self).__init__(*args)

    def forward(self, batch):
        embed = self.embed(batch.enc_src.id)
        self.code = code = self.enc(embed, batch.enc_src.len)
        decoded = self.dec.teacher_forcing(code, batch)
        loss, acc = self._loss_and_accuracy(
            decoded.prob, batch.dec_tar.id, self.embed.vocab_size)
        return decoded
