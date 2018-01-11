import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import to_gpu


class SampleDiscriminator(nn.Module):
    def __init__(self, max_len, filter_size, step_dim, embed_size, dropout):
        super(SampleDiscriminator, self).__init__()
        # Sentence should be represented as a matrix : [max_len x step_size]
        # Step represetation can be :
        #   - hidden states which each word is generated from
        #   - embeddings that were porduced from generated word indices
        # inputs.size() : [batch_size, 1, max_len, embed_size]

        def map_len(in_size, stride):
            return (in_size - filter_size) // stride + 1
        out1_len = map_len(max_len, 1)
        out2_len = map_len(out1_len, 2)
        out3_len = map_len(out2_len, 2)
        print('conv3_len:', out3_len)

        ch = [128, 256, 512]

        #self.embedding = nn.Embedding(ntokens, embed_size)
        #self.convs1 = [nn.Conv2d(Ci, Co, (K, D)) for K in Ks]
        self.conv1 = nn.Conv2d(1, ch[0], (filter_size, step_dim)) # H : max_len
        self.conv2 = nn.Conv2d(ch[0], ch[1], (filter_size, 1), stride=2)
        self.conv3 = nn.Conv2d(ch[1], ch[2], (filter_size, 1), stride=2)
        self.pool = nn.MaxPool2d((out3_len, 1))
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(ch[2], 1)
        self.attn1_1 = nn.Linear(ch[0], (ch[0]//2))
        self.attn1_2 = nn.Linear((ch[0]//2), 1)
        self.attn2_1 = nn.Linear(ch[1], ch[1]//2)
        self.attn2_2 = nn.Linear(ch[1]//2, 1)
        self.attn3_1 = nn.Linear(ch[2], ch[2]//2)
        self.attn3_2 = nn.Linear(ch[2]//2, 1)

    def forward(self, inputs):
        inputs = inputs.unsqueeze(1) # [batch_size, 1, max_len, embed_size]
        attn_weights = []

        out1 = F.relu(self.conv1(inputs)) # [batch_size, 128, 18, 1]
        attn1 = out1.view(out1.size(0)*out1.size(2), out1.size(1)) # [batch_size*18, 128]
        attn1 = F.relu(self.attn1_1(attn1)) # [batch_size*18, 64]
        attn1 = F.relu(self.attn1_2(attn1)) # [batch_size*18, 1]
        attn1 = attn1.view(out1.size(0), out1.size(2))
        attn1 = F.softmax(attn1, dim=1) # [batch_size, 18]
        attn_weights.append(attn1.data.cpu().numpy())
        attn1 = attn1.contiguous().view(attn1.size(0), 1, attn1.size(1), 1)
        attn_out1 = out1 * attn1.expand_as(out1)
        out1 = out1 + attn_out1

        out2 = F.relu(self.conv2(out1)) # [batch_size, 256, 8, 1]
        attn2 = out2.view(out2.size(0)*out2.size(2), out2.size(1))
        attn2 = F.relu(self.attn2_1(attn2))
        attn2 = F.relu(self.attn2_2(attn2))
        attn2 = attn2.view(out2.size(0), out2.size(2))
        attn2 = F.softmax(attn2, dim=1) # [batch_size, 18]
        attn_weights.append(attn2.data.cpu().numpy())
        attn2 = attn2.contiguous().view(attn2.size(0), 1, attn2.size(1), 1)
        attn_out2 = out2 * attn2.expand_as(out2)
        out2 = out2 + attn_out2

        out3 = F.relu(self.conv3(out2)) # [batch_size, 512, 3, 1]
        attn3 = out3.view(out3.size(0)*out3.size(2), out3.size(1))
        attn3 = F.relu(self.attn3_1(attn3))
        attn3 = F.relu(self.attn3_2(attn3))
        attn3 = attn3.view(out3.size(0), out3.size(2))
        attn3 = F.softmax(attn3, dim=1) # [batch_size, 18]
        attn_weights.append(attn3.data.cpu().numpy())
        attn3 = attn3.contiguous().view(attn3.size(0), 1, attn3.size(1), 1)
        attn_out3 = out3 * attn3.expand_as(out3)
        out3 = out3 + attn_out3

        out4 = self.pool(out3) # [batch_size, 512, 1, 1]
        out5 = self.dropout(out4)
        out6 = self.fc1(out5.squeeze()) # [batch_size, 512]
        out7 = torch.mean(out6) # [batch_size]

        return out7, attn_weights

    @staticmethod
    def train_(cfg, disc, real_states, fake_states):
            # clamp parameters to a cube
        for p in disc.parameters():
            p.data.clamp_(-cfg.gan_clamp, cfg.gan_clamp) # [min,max] clamp
            # WGAN clamp (default:0.01)

        disc.train()
        disc.zero_grad()

        # loss / backprop
        err_d_real, attn_real = disc(real_states.detach())
        one = to_gpu(cfg.cuda, torch.FloatTensor([1]))
        err_d_real.backward(one)

        # negative samples ----------------------------
        # loss / backprop
        err_d_fake, attn_fake = disc(fake_states.detach())
        err_d_fake.backward(one * -1)

        err_d = -(err_d_real - err_d_fake)

        return ((err_d.data[0], err_d_real.data[0], err_d_fake.data[0]),
               (attn_real, attn_fake))
