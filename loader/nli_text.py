"""Modifed from https://github.com/jakezhaojb/ARAE/tree/master/pytorch"""

import argparse
import codecs
import json
import os
import zipfile


"""
Transforms SNLI & MultiNLI data into lines of text files.
Gets rid of repeated premise sentences.
"""

SNLI_URL = "https://nlp.stanford.edu/projects/snli/snli_1.0.zip"
MNLI_URL = "https://www.nyu.edu/projects/bowman/multinli/multinli_1.0.zip"


def download_and_unzip(url, outdir):
    zip = os.path.join(outdir, url.split('/')[-1])
    os.system("wget -O {0} {1}".format(zip, url))

    with zipfile.ZipFile(zip, 'r') as zip:
        print('Unzipping %s..' % zip)
        zip.extractall(outdir)
        print('Done!')


def transform_data(in_path):
    print("Loading", in_path)

    premises = []
    hypotheses = []

    last_premise = None
    with codecs.open(in_path, encoding='utf-8') as f:
        for line in f:
            loaded_example = json.loads(line)

            # load premise
            raw_premise = loaded_example['sentence1_binary_parse'].split(" ")
            premise_words = []
            # loop through words of premise binary parse
            for word in raw_premise:
                # don't add parse brackets
                if word != "(" and word != ")":
                    premise_words.append(word)
            premise = " ".join(premise_words)

            # load hypothesis
            raw_hypothesis = \
                loaded_example['sentence2_binary_parse'].split(" ")
            hypothesis_words = []
            for word in raw_hypothesis:
                if word != "(" and word != ")":
                    hypothesis_words.append(word)
            hypothesis = " ".join(hypothesis_words)

            # make sure to not repeat premiess
            if premise != last_premise:
                premises.append(premise)
            hypotheses.append(hypothesis)

            last_premise = premise

    return premises, hypotheses


def write_sentences(write_path, premises, hypotheses, append=False):
    print("Writing to {}\n".format(write_path))
    if append:
        with open(write_path, "a") as f:
            for p in premises:
                f.write(p)
                f.write("\n")
            for h in hypotheses:
                f.write(h)
                f.write('\n')
    else:
        with open(write_path, "w") as f:
            for p in premises:
                f.write(p)
                f.write("\n")
            for h in hypotheses:
                f.write(h)
                f.write('\n')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, default="../data/nli",
                        help='path to snli data')
    args = parser.parse_args()

    if not os.path.exists(args.path):
        os.makedirs(args.path)
        print("Creating directory "+args.path)

    if not os.path.exists(os.path.join(args.path, SNLI_URL.split('/')[-1])):
        download_and_unzip(SNLI_URL, args.path)

    if not os.path.exists(os.path.join(args.path, MNLI_URL.split('/')[-1])):
        download_and_unzip(MNLI_URL, args.path)

    # process and write test.txt and train.txt files
    # SNLI
    snli_path = os.path.join(args.path, "snli_1.0", "snli_1.0_")
    premises, hypotheses = transform_data(snli_path + "test.jsonl"))
    write_sentences(write_path=os.path.join(args.path, "test.txt"),
                    premises=premises, hypotheses=hypotheses)

    premises, hypotheses = transform_data(snli_path + "train.jsonl"))
    write_sentences(write_path=os.path.join(args.path, "train.txt"),
                    premises=premises, hypotheses=hypotheses)

    premises, hypotheses = transform_data(snli_path + "dev.jsonl"))
    write_sentences(write_path=os.path.join(args.path, "train.txt"),
                    premises=premises, hypotheses=hypotheses, append=True)

    # MNLI
    mnli_path = os.path.join(args.path, "multinli_1.0", "multinli_1.0_")
    premises, hypotheses = transform_data(mnli_path + "dev_matched.jsonl")
    write_sentences(write_path=os.path.join(args.path, "test.txt"),
                    premises=premises, hypotheses=hypotheses, append=True)

    premises, hypotheses = transform_data(mnli_path + "train.jsonl"))
    write_sentences(write_path=os.path.join(args.path, "train.txt"),
                    premises=premises, hypotheses=hypotheses, append=True)
