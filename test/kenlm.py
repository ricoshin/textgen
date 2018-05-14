"""Modifed from https://github.com/jakezhaojb/ARAE/tree/master/pytorch"""

import os


def load_kenlm():
    global kenlm
    import kenlm


def train_kenlm(net, decoded_text, global_step):
    save_name = "generated_{}".format(global_step)
    save_dir = os.path.join(net.cfg.log_dir, 'kenlm')
    eval_path = os.path.join(net.cfg.data_dir, 'test.txt')
    data_path = os.path.join(save_dir, save_name+".txt")
    output_path = os.path.join(save_dir, save_name+".arpa")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    with open(data_path, 'w') as f:
        # laplacian smoothing
        for word in net.vocab_w.word2idx.keys():
            f.write(word+"\n")

        for text in decoded_text:
            f.write(text+"\n")

    # train language model on generated examples

    lm = train_ngram_lm(kenlm_path=net.cfg.kenlm_path,
                        data_path=data_path,
                        output_path=output_path,
                        N=net.cfg.N)

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
    data_path = os.path.join(curdir, data_path)
    output_path = os.path.join(curdir, output_path)
    command = "bin/lmplz --skip_symbols -S 10% -o "+str(N)+" <"+\
              data_path + " >" + output_path
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
