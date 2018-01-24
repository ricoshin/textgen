from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.bleu_score import modified_precision
from test.evaluate_nltk import truncate
"""
This function is mainly from LeakGAN eval_bleu.py
url https://github.com/CR-Gjx/LeakGAN/blob/master/Image%20COCO/eval_bleu.py
some parts are modified
"""
def leakgan_bleu(references, hypotheses):
    ref = [truncate(s) for s in references]
    hyp = [truncate(s) for s in hypotheses]
    result = [0, 0, 0]
    for ngram in range(2,4):
        weight = tuple((1. / ngram for _ in range(ngram)))
        bleu = []
        num = 0
        for h in hyp:
            bleu.append(sentence_bleu(ref, h, weight))
            num += 1
        result[ngram-2] = 1.0 * sum(bleu) / len(bleu)
    return result

"""
This function is mainly from UROP-Adversarial-Feature-Matching-for-Text-Generation
url https://github.com/Jeff-HOU/UROP-Adversarial-Feature-Matching-for-Text-Generation/blob/master/code/utils.py
some parts are modified
"""
def urop_bleu(references, hypotheses):
    ref = [truncate(s) for s in references]
    hyp = [truncate(s) for s in hypotheses]
    result = [0.] * 3 # 3 = len([2,3,4])
    for ngram in range(2, 4):
        for s in hyp: # sentence-wise bleu
 	        result[ngram-2] += round(modified_precision(ref, s, n=ngram), 5)
    result = [x / len(hyp) for x in result]
    return result

