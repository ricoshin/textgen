# textgen QA branch
This branch is built for question answering.

## Dependencies
* common : spacy, pytorch, tensorboardX, tqdm and more

## Preparing Data
* Download glove and store in proper path

### Simple Questions Dataset

## Using Test Session
* Pass argument `--test` when running main python file and specify the trained data stored in `out/` directory by passing argument `--name`
* Full evluation additionally performs bleu-1, bleu-2, bleu-3, default nltk bleu(mixed ngram evaluation)
* Simple evaluation performs bleu-4 and perplexity evluation.

## Additional parser description
arg | default | description
-----|---------|------------
`--data_name` | simple_questions | 'snli' and 'BookCorpus' are not available
