

def train_lm(eval_path, save_path):
    # ppl = train_lm(eval_path=os.path.join(args.data_path, "test.txt"),
    #                save_path="./output/{}/end_of_epoch{}_lm_generations".
    #                format(args.outf, epoch))

    # generate examples
    indices = []
    noise = to_gpu(args.cuda, Variable(torch.ones(100, args.z_size)))
    # noise.size() : [100 x z_size]
    for i in range(1000):
        # 1000 batches size of 100
        noise.data.normal_(0, 1)

        fake_hidden = gan_gen(noise)
        max_indices = autoencoder.generate(fake_hidden, args.maxlen)
        # max_indices.size() : []
        indices.append(max_indices.data.cpu().numpy())

    indices = np.concatenate(indices, axis=0)
    # generated indices from fake hidden state (100 x 1000 sentences)

    # write generated sentences to text file
    with open(save_path+".txt", "w") as f:
        # laplacian smoothing
        for word in self.net.vocab_w.word2idx.keys():
            f.write(word+"\n")
        for idx in indices:
            # generated sentence
            words = [corpus.dictionary.idx2word[x] for x in idx]
            # truncate sentences to first occurrence of <eos>
            truncated_sent = []
            for w in words:
                if w != '<eos>':
                    truncated_sent.append(w)
                else:
                    break
            chars = " ".join(truncated_sent)
            f.write(chars+"\n")

    # train language model on generated examples
    lm = train_ngram_lm(kenlm_path=args.kenlm_path,
                        data_path=save_path+".txt",
                        output_path=save_path+".arpa",
                        N=args.N)

    # load sentences to evaluate on
    with open(eval_path, 'r') as f:
        lines = f.readlines()
    sentences = [l.replace('\n', '') for l in lines]
    ppl = get_ppl(lm, sentences)

    return ppl


def train_ngram_lm(kenlm_path, data_path, output_path, N):
    """
    Trains a modified Kneser-Ney n-gram KenLM from a text file.
    Creates a .arpa file to store n-grams.
    """
    # create .arpa file of n-grams
    curdir = os.path.abspath(os.path.curdir)
    #
    command = "bin/lmplz -o "+str(N)+" <"+os.path.join(curdir, data_path) + \
              " >"+os.path.join(curdir, output_path)
    os.system("cd "+os.path.join(kenlm_path, 'build')+" && "+command)

    load_kenlm()
    # create language model
    model = kenlm.Model(output_path)

    return model


def get_ppl(lm, sentences):
    """
    Assume sentences is a list of strings (space delimited sentences)
    """
    total_nll = 0
    total_wc = 0
    for sent in sentences:
        words = sent.strip().split()
        score = lm.score(sent, bos=True, eos=False)
        word_count = len(words)
        total_wc += word_count
        total_nll += score
    ppl = 10**-(total_nll/total_wc)
    return ppl
