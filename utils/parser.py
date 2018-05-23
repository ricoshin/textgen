import argparse

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

parser = argparse.ArgumentParser(description='PyTorch ARAE for Text')
# Path Arguments
parser.add_argument('--prepro_dir', type=str, default='prepro',
                    help='location of the preprocessed data')
parser.add_argument('--data_dir', type=str, default='data',
                    help='location of the datasets')
parser.add_argument('--data_name', type=str, default='nli',
                    choices=['nli','books', 'pos', 'snli'], help='name of dataset')
parser.add_argument('--glove_dir', type=str, default='data/glove',
                    help='location of pretrained glove data')
parser.add_argument('--out_dir', type=str, default='out2',
                    help='location of output files')
parser.add_argument('--name', type=str, required=True)
parser.add_argument('--kenlm_path', type=str, default='kenlm',
                    help='path to kenlm directory')

# Data Processing Arguments

parser.add_argument('--min_len', type=int, default=1,
                    help='minimum sentence length')
parser.add_argument('--max_len', type=int, default=20,
                    help='maximum sentence length')
parser.add_argument('--exclude_over_max', type=str2bool, default=True,
                    help='exclude from dataset if sent len is over max_len')
#parser.add_argument('--lowercase', action='store_true',
#                    help='lowercase all text')
parser.add_argument('--reload_prepro', action='store_true')
parser.add_argument('--load_glove', type=str2bool, default=True,
                    help='initialize embedding matrix using glove')

# Model Arguments
parser.add_argument('--vocab_size_w', type=int, default=20000,
                    help='cut vocabulary down to this size ')
parser.add_argument('--embed_size_w', type=int, default=300,
                    help='size of word embeddings')
parser.add_argument('--embed_size_t', type=int, default=50,
                    help='size of tag embeddings')
parser.add_argument('--hidden_size_w', type=int, default=300,
                    help='number of decoder hidden units per layer')
parser.add_argument('--hidden_size_t', type=int, default=50,
                    help='number of tagger hidden units per layer')
parser.add_argument('--nlayers', type=int, default=1,
                    help='number of layers')
parser.add_argument('--noise_radius', type=float, default=0.0,
                    help='stdev of noise for autoencoder (regularizer)')
parser.add_argument('--noise_anneal', type=float, default=0.995,
                    help='anneal noise_radius exponentially by this'
                         'every 100 iterations')
parser.add_argument('--code_norm', type=str2bool, default=False,
                    help='encoder code normalization')
parser.add_argument('--hidden_init', action='store_true',
                    help="initialize decoder hidden state with encoder's")
parser.add_argument('--arch_g', type=str, default='300-300',
                    help='generator architecture (MLP)')
parser.add_argument('--arch_d', type=str, default='300-300',
                    help='critic/discriminator architecture (MLP)')
parser.add_argument('--z_size', type=int, default=100,
                    help='dimension of random noise z to feed into generator')
parser.add_argument('--temp', type=float, default=1,
                    help='softmax temperature (lower --> more discrete)')
parser.add_argument('--ae_grad_norm', type=str2bool, default=True,
                    help='norm code gradient from critic->encoder')
parser.add_argument('--gan_to_enc', type=float, default=-1.0,
                    help='weight factor passing gradient from gan to encoder')
parser.add_argument('--gan_to_dec', type=float, default=1.0,
                    help='weight factor passing gradient from gan to decoder')
parser.add_argument('--dropout', type=float, default=0.0,
                    help='dropout applied to layers (0 = no dropout)')
parser.add_argument('--kernel_sizes', type=str, default='2,3,4',
                    help='kernel sizes of text CNN')
parser.add_argument('--kernel_num', type=int, default=100,
                    help='number of each size of kernel')
parser.add_argument('--with_attn', type=str2bool, default=False,
                    help='including n-gram attention discriminator')
parser.add_argument('--disc_s_in', type=str, default='embed',
                    choices=['embed', 'hidden', 'both'],
                    help='disc_s input type')
parser.add_argument('--enc_disc', type=str2bool, default=True,
                    help='weight sharing between encoder and disc_s')
parser.add_argument('--pos_tag', type=str2bool, default=False,
                    help='determine whether the model use POS tags')
parser.add_argument('--enc_type', type=str, default='cnn',
                    choices=['cnn','rnn'], help='encoder type (CNN or RNN)')
parser.add_argument('--dec_type', type=str, default='rnn',
                    choices=['cnn','rnn'], help='encoder type (CNN or RNN)')
parser.add_argument('--dec_embed', type=str2bool, default=False,
                    help='decoder outputs word embeddings instead of indices')

# Training Arguments
parser.add_argument('--kl_term', type=float, default=0.01,
                    help='kl term coefficient')
parser.add_argument('--epochs', type=int, default=15,
                    help='maximum number of epochs')
parser.add_argument('--min_epochs', type=int, default=6,
                    help="minimum number of epochs to train for")
parser.add_argument('--no_earlystopping', action='store_true',
                    help="won't use KenLM for early stopping")
parser.add_argument('--patience', type=int, default=5,
                    help="number of language model evaluations without ppl "
                         "improvement to wait before early stopping")
parser.add_argument('--batch_size', type=int, default=64, metavar='N',
                    help='batch size')
parser.add_argument('--eval_size', type=int, default=500, metavar='N',
                    help='batch size during evaluation')
parser.add_argument('--niter_ae', type=int, default=1,
                    help='number of autoencoder iterations in training')
parser.add_argument('--niter_gan_d', type=int, default=5,
                    help='number of discriminator iterations in training')
parser.add_argument('--niter_gan_g', type=int, default=1,
                    help='number of generator iterations in training')
parser.add_argument('--niter_gan_schedule', type=str, default='2-4-6',
                    help='epoch counts to increase number of GAN training '
                         ' iterations (increment by 1 each time)')
parser.add_argument('--lr_ae', type=float, default=1,
                    help='autoencoder learning rate')
parser.add_argument('--lr_gan_g', type=float, default=5e-05,
                    help='generator learning rate')
parser.add_argument('--lr_gan_d', type=float, default=1e-05,
                    help='critic/discriminator learning rate')
parser.add_argument('--beta1', type=float, default=0.9,
                    help='beta1 for adam. default=0.9')
parser.add_argument('--clip', type=float, default=1,
                    help='gradient clipping, max norm')
parser.add_argument('--gan_clamp', type=float, default=0.01,
                    help='WGAN clamp')
parser.add_argument('--backprop_gen', type=str2bool, default=False,
                    help='enable backpropagation gradient from disc_s to gen')
parser.add_argument('--disc_s_hold', type=int, default=15,
                    help='num of initial epochs not training train disc_s')
parser.add_argument('--fix_embed', type=str2bool, default=False,
                    help='pretain embedding matrix weights (not trainable)')
parser.add_argument('--word_temp', type=float, default=1e-2,
                    help='softmax temperature for wordwise attention')
parser.add_argument('--layer_temp', type=float, default=1e-2,
                    help='softmax temperature for layerwise attention')
parser.add_argument('--anneal_step', type=int, default=200,
                    help='autoencdoer noise annealing interval')
parser.add_argument('--embed_temp', type=float, default=200,
                    help='temperature of log softmax in word prediction')

# Evaluation Arguments
parser.add_argument('--sample', action='store_true',
                    help='sample when decoding for generation')
parser.add_argument('--N', type=int, default=5,
                    help='N-gram order for training n-gram language model')
parser.add_argument('--log_interval', type=int, default=50,
                    help='interval to log autoencoder training results')

# Test Arguments
#parser.add_argument('--test', type=bool, default=False, help='pass True to enter test session')

# Other
parser.add_argument('--small', action='store_true') # just for debugging
parser.add_argument('--log_level', type=str, default='debug')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--cuda', type=str2bool, default=True, help='use CUDA')
parser.add_argument('--log_nsample', type=int, default=4)
parser.add_argument('--test', action='store_true', help='run test mode')
parser.add_argument('--visualize', action='store_true',
                    help='run visualize mode')
