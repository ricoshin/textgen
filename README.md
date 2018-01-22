# textgen metric branch
This branch is built for evaluation including bleu and perplextiy.
For bleu, nltk module is implemented.
For perplexity, kenlm module is implemented.

## Dependencies
nltk, kenlm

## Preparing Data
* Download glove and store in proper path

### SNLI Data
* Follow the instruction in ARAE/pytorch `https://github.com/jakezhaojb/ARAE/tree/master/pytorch`

### BookCorpus

## Building Kenlm
* Clone kenlm repo https://github.com/kpu/kenlm
* Compile kenlm, following the instruction in kenlm.
* Find `train_lm()` function and set `kenlm_path` variable to proper directory. For example, `kenlm_path = '/home/username/kenlm'`
* For possible import problem, install kenlm via pip: `pip install kenlm`

## About NLTK Bleu
It is possible that bleu score implemented in this repo could be wrong.

