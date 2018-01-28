# textgen metric branch
This branch is built for evaluation including bleu and perplextiy.
For bleu, nltk module is implemented.
For perplexity, kenlm module is implemented.

## Dependencies
* common : spacy, pytorch, tensorboardX, tqdm and more
* in this branch : nltk, kenlm, boost(kenlm dependency)

## Preparing Data
* Download glove and store in proper path

### SNLI Data
* Follow the instruction in ARAE/pytorch `https://github.com/jakezhaojb/ARAE/tree/master/pytorch`

### BookCorpus

## Building Kenlm
* Clone kenlm repo https://github.com/kpu/kenlm
* Compile kenlm, following the instruction in kenlm.
* Find `train_lm()` function in `train_with_kenlm.py` and set `kenlm_path` variable to proper directory. For example, `kenlm_path = '/home/username/kenlm'`
* install kenlm by running `python setup.py install` in kenlm directory.

## About NLTK Bleu
It is possible that bleu score implemented in this repo could be wrong.

## Using Test Session
* Pass argument `--test` when running main python file and specify the trained data stored in `out/` directory by passing argument `--name`
* Full evluation additionally performs bleu-1, bleu-2, bleu-3, default nltk bleu(mixed ngram evaluation)
* Simple evaluation performs bleu-4 and perplexity evluation.

## Additional parser description
arg | default | description
-----|---------|------------
`--word_act` | softmax, sigmoid, sparsemax | choose activation function in word attention
`--test` | | run test session using trained data
