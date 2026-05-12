"""
UniEval baseline (Zhong et al., 2022) — wrapper around official implementation.

Delegates to the evaluators in unieval_official.py (SumEvaluator, D2tEvaluator,
FactEvaluator, TranslationEvaluator) which use Bool-QA formatted prompts with
the MingZhong/unieval-sum (or unieval-fact) T5 checkpoint.

The official implementation handles dimension-specific prompt construction
(add_question), sentence-level vs. summary-level aggregation, and
Yes/No probability scoring.
"""

import numpy as np
from typing import List, Dict, Optional, Union
import logging

from .unieval_official import (
    SumEvaluator,
    D2tEvaluator,
    FactEvaluator,
    TranslationEvaluator,
    UniEvaluator,
    convert_to_json,
    add_question,
    get_evaluator,
)

logger = logging.getLogger(__name__)

TASK_DIMENSIONS = {
    "summarization": ["coherence", "consistency", "fluency", "relevance"],
    "data2text": ["naturalness", "informativeness"],
    "fact": ["consistency"],
    "translation": ["fluency", "accuracy"],
}


class UniEvalScorer:
    """UniEval wrapper compatible with the DiffScore baseline interface.

    Uses the official UniEval evaluators from unieval_official.py.
    Supports per-dimension scoring and overall (average) scoring.
    """

    def __init__(
        self,
        task: str = "summarization",
        device: str = "cuda",
        max_length: int = 1024,
        batch_size: int = 8,
        cache_dir: Optional[str] = None,
    ):
        self.task = task
        self.device = device
        self.batch_size = batch_size
        self._evaluator = get_evaluator(
            task, max_length=max_length, device=device, cache_dir=cache_dir
        )
        self.dimensions = TASK_DIMENSIONS.get(
            task, ["coherence", "fluency"]
        )
        logger.info(f"UniEvalScorer initialized: task={task}, dims={self.dimensions}")

    def score(
        self,
        sources: List[str],
        candidates: List[str],
        references: Optional[List[str]] = None,
        dimensions: Optional[List[str]] = None,
    ) -> Dict[str, np.ndarray]:
        """Score texts across specified dimensions using the official evaluator.

        Returns:
            Dict mapping "unieval_{dim}" -> (N,) scores, plus "unieval_overall".
        """
        dims = dimensions or self.dimensions

        if self.task == "summarization":
            return self._score_summarization(sources, candidates, references, dims)
        elif self.task == "data2text":
            return self._score_d2t(candidates, references, dims)
        elif self.task == "fact":
            return self._score_fact(sources, candidates)
        elif self.task == "translation":
            return self._score_translation(sources, candidates, references, dims)
        else:
            raise ValueError(f"Unsupported task: {self.task}")

    def _score_summarization(
        self,
        sources: List[str],
        candidates: List[str],
        references: Optional[List[str]],
        dims: List[str],
    ) -> Dict[str, np.ndarray]:
        n = len(candidates)
        ref_list = references if references else ["" for _ in range(n)]
        data = convert_to_json(
            output_list=candidates,
            src_list=sources,
            ref_list=ref_list,
        )

        evaluator: SumEvaluator = self._evaluator
        eval_scores = [{} for _ in range(n)]

        for dim in dims:
            if dim == "consistency" or dim == "fluency":
                from nltk import sent_tokenize
                src_list, output_list = [], []
                n_sents = []
                for i in range(n):
                    src = data[i]["source"] if dim == "consistency" else ""
                    sents = sent_tokenize(data[i]["system_output"])
                    n_sents.append(len(sents))
                    for s in sents:
                        src_list.append(src)
                        output_list.append(s)
                input_list = add_question(
                    dimension=dim, output=output_list,
                    src=src_list, task="summarization",
                )
                sent_score = evaluator.scorer.score(input_list, batch_size=self.batch_size)
                start_idx = 0
                for i, cnt in enumerate(n_sents):
                    eval_scores[i][dim] = sum(sent_score[start_idx:start_idx + cnt]) / cnt
                    start_idx += cnt

            elif dim == "coherence" or dim == "relevance":
                src_list, output_list, ref_l = [], [], []
                for i in range(n):
                    src_list.append(data[i]["source"])
                    output_list.append(data[i]["system_output"])
                    if dim == "relevance":
                        ref_l.append(data[i].get("reference", ""))
                input_list = add_question(
                    dimension=dim, output=output_list,
                    src=src_list, ref=ref_l if ref_l else None,
                    task="summarization",
                )
                score = evaluator.scorer.score(input_list, batch_size=self.batch_size)
                for i in range(n):
                    eval_scores[i][dim] = score[i]

        return self._format_results(eval_scores, dims)

    def _score_d2t(
        self,
        candidates: List[str],
        references: Optional[List[str]],
        dims: List[str],
    ) -> Dict[str, np.ndarray]:
        n = len(candidates)
        ref_list = references if references else ["" for _ in range(n)]
        data = convert_to_json(output_list=candidates, ref_list=ref_list)

        evaluator: D2tEvaluator = self._evaluator
        eval_scores = [{} for _ in range(n)]

        for dim in dims:
            output_list, ref_l = [], []
            for i in range(n):
                output_list.append(data[i]["system_output"])
                ref_l.append(data[i].get("reference", ""))
            input_list = add_question(
                dimension=dim, output=output_list,
                ref=ref_l, task="data2text",
            )
            score = evaluator.scorer.score(input_list, batch_size=self.batch_size)
            for i in range(n):
                eval_scores[i][dim] = score[i]

        return self._format_results(eval_scores, dims)

    def _score_fact(
        self,
        sources: List[str],
        candidates: List[str],
    ) -> Dict[str, np.ndarray]:
        n = len(candidates)
        data = convert_to_json(output_list=candidates, src_list=sources)
        overall = self._evaluator.evaluate(data)
        return {
            "unieval_consistency": overall,
            "unieval_overall": overall,
        }

    def _score_translation(
        self,
        sources: List[str],
        candidates: List[str],
        references: Optional[List[str]],
        dims: List[str],
    ) -> Dict[str, np.ndarray]:
        n = len(candidates)
        ref_list = references if references else sources
        data = convert_to_json(
            output_list=candidates, src_list=sources, ref_list=ref_list,
        )
        overall = self._evaluator.evaluate(data, dims=dims)

        evaluator: TranslationEvaluator = self._evaluator
        eval_scores = [{} for _ in range(n)]

        for dim in dims:
            if dim == "fluency":
                from nltk import sent_tokenize
                output_list, n_sents = [], []
                for i in range(n):
                    sents = sent_tokenize(data[i]["system_output"])
                    n_sents.append(len(sents))
                    output_list.extend(sents)
                src_list = ["" for _ in range(len(output_list))]
                input_list = add_question(
                    dimension=dim, output=output_list,
                    src=src_list, task="translation",
                )
                sent_score = evaluator.scorer.score(input_list, batch_size=self.batch_size)
                start_idx = 0
                for i, cnt in enumerate(n_sents):
                    eval_scores[i][dim] = sum(sent_score[start_idx:start_idx + cnt]) / cnt
                    start_idx += cnt
            elif dim == "accuracy":
                src_list, output_list, ref_l = [], [], []
                for i in range(n):
                    src_list.append(data[i].get("source", ""))
                    output_list.append(data[i]["system_output"])
                    ref_l.append(data[i].get("reference", ""))
                input_list = add_question(
                    dimension=dim, output=output_list,
                    src=src_list, ref=ref_l, task="translation",
                )
                score = evaluator.scorer.score(input_list, batch_size=self.batch_size)
                for i in range(n):
                    eval_scores[i][dim] = score[i]

        return self._format_results(eval_scores, dims)

    def _format_results(
        self, eval_scores: List[Dict], dims: List[str]
    ) -> Dict[str, np.ndarray]:
        results = {}
        for dim in dims:
            results[f"unieval_{dim}"] = np.array(
                [s.get(dim, 0.0) for s in eval_scores]
            )
        all_dim_arrays = [results[f"unieval_{d}"] for d in dims if f"unieval_{d}" in results]
        if all_dim_arrays:
            results["unieval_overall"] = np.mean(all_dim_arrays, axis=0)
        return results

    def score_overall(
        self,
        sources: List[str],
        candidates: List[str],
        references: Optional[List[str]] = None,
    ) -> np.ndarray:
        """Return only the overall (averaged) score."""
        results = self.score(sources, candidates, references)
        return results.get("unieval_overall", np.zeros(len(candidates)))
