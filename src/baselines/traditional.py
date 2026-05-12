"""
Traditional NLG evaluation metrics: BLEU, ROUGE, METEOR.
"""

import numpy as np
from typing import List, Dict, Optional, Union
import logging

logger = logging.getLogger(__name__)


class BLEUScorer:
    """BLEU score computation using sacrebleu."""

    def score(
        self, candidates: List[str], references: Union[List[str], List[List[str]]]
    ) -> np.ndarray:
        import sacrebleu

        scores = []
        if isinstance(references[0], list):
            for cand, ref in zip(candidates, references):
                bleu = sacrebleu.sentence_bleu(cand, ref)
                scores.append(bleu.score / 100.0)
            return np.array(scores)
        else:
            for cand, ref in zip(candidates, references):
                bleu = sacrebleu.sentence_bleu(cand, [ref])
                scores.append(bleu.score / 100.0)
            return np.array(scores)

    def score_corpus(
        self, candidates: List[str], references: List[str]
    ) -> float:
        import sacrebleu
        if isinstance(references[0], list):
            bleu = sacrebleu.corpus_bleu(candidates, references)
        else:
            bleu = sacrebleu.corpus_bleu(candidates, [references])
        return bleu.score / 100.0


class ROUGEScorer:
    """ROUGE score computation."""

    def __init__(self):
        from rouge_score import rouge_scorer
        self.scorer = rouge_scorer.RougeScorer(
            ["rouge1", "rouge2", "rougeL"], use_stemmer=True
        )

    def score(
        self, candidates: List[str], references: Union[List[str], List[List[str]]]
    ) -> Dict[str, np.ndarray]:
        results = {"rouge1": [], "rouge2": [], "rougeL": []}

        if isinstance(references[0], list):
            ## multi reference
            for cand, ref in zip(candidates, references):
                scores_list = [self.scorer.score(ref_single, cand) for ref_single in ref]
                for key in results:
                    score_list = [scores[key].fmeasure for scores in scores_list]
                    results[key].append(sum(score_list) / len(score_list))

            return {k: np.array(v) for k, v in results.items()}
        else:
            for cand, ref in zip(candidates, references):
                scores = self.scorer.score(ref, cand)
                for key in results:
                    results[key].append(scores[key].fmeasure)

            return {k: np.array(v) for k, v in results.items()}


class METEORScorer:
    """METEOR score computation using NLTK."""

    def __init__(self):
        import nltk
        try:
            nltk.data.find("corpora/wordnet")
        except LookupError:
            nltk.download("wordnet", quiet=True)
            nltk.download("punkt_tab", quiet=True)

    def score(
        self, candidates: List[str], references: Union[List[str], List[List[str]]]
    ) -> np.ndarray:
        from nltk.translate.meteor_score import meteor_score, single_meteor_score
        from nltk.tokenize import word_tokenize

        scores = []
        if isinstance(references[0], List):
            for cand, ref in zip(candidates, references):
                try:
                    ref_token_list = [word_tokenize(single_ref) for single_ref in ref]
                    s = meteor_score(ref_token_list, word_tokenize(cand))
                except Exception:
                    s = 0.0
                scores.append(s)
            return np.array(scores)
        else:
            for cand, ref in zip(candidates, references):
                try:
                    s = single_meteor_score(word_tokenize(ref), word_tokenize(cand))
                except Exception:
                    s = 0.0
                scores.append(s)
            return np.array(scores)
