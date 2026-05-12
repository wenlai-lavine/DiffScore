"""
QuestEval baseline (Scialom et al., 2021).

QA-based evaluation that measures consistency by generating questions from
the source/candidate and checking if answers match. Uses pre-trained
question generation and question answering models.
"""

import numpy as np
from typing import List, Dict, Optional
import logging

logger = logging.getLogger(__name__)


class QuestEvalScorer:
    """QuestEval: QA-based factual consistency evaluation.

    Wraps the questeval package. If unavailable, falls back to an
    NLI-based approximation using a lightweight entailment model.
    """

    def __init__(
        self,
        task: str = "summarization",
        device: str = "cuda",
        batch_size: int = 16,
    ):
        self.task = task
        self.device = device
        self.batch_size = batch_size
        self._questeval = None

        try:
            from questeval.questeval_metric import QuestEval
            self._questeval = QuestEval(
                # task=task,
                do_weighter=True,
                no_cuda=False,
            )
            logger.info("Using questeval package")
        except (ImportError, Exception) as e:
            logger.warning(
                f"questeval package not available ({e}). "
                "Using NLI-based fallback. Install: pip install questeval"
            )
            self._setup_fallback()

    def _setup_fallback(self):
        """Set up NLI-based fallback for QA-style evaluation."""
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        model_name = "cross-encoder/nli-deberta-v3-base"
        self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        self._model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self._model.eval()
        self._model.to(self.device)

    def score(
        self,
        sources: List[str],
        candidates: List[str],
    ) -> np.ndarray:
        """Compute QuestEval scores."""
        if self._questeval is not None:
            return self._score_questeval(sources, candidates)
        return self._score_fallback(sources, candidates)

    def _score_questeval(
        self, sources: List[str], candidates: List[str]
    ) -> np.ndarray:
        """Score using the official questeval package."""
        result = self._questeval.corpus_questeval(
            hypothesis=candidates,
            sources=sources,
        )
        return np.array(result["ex_level_scores"])

    def _score_fallback(
        self, sources: List[str], candidates: List[str]
    ) -> np.ndarray:
        """NLI-based fallback: measure bidirectional entailment."""
        import torch
        import torch.nn.functional as F

        scores = []
        for i in range(0, len(sources), self.batch_size):
            batch_src = sources[i:i + self.batch_size]
            batch_cand = candidates[i:i + self.batch_size]

            # Forward: source entails candidate
            fwd_enc = self._tokenizer(
                batch_src, batch_cand,
                return_tensors="pt", padding=True,
                truncation=True, max_length=512,
            ).to(self.device)

            with torch.no_grad():
                fwd_out = self._model(**fwd_enc)
                fwd_probs = F.softmax(fwd_out.logits, dim=-1)
                fwd_ent = fwd_probs[:, -1]  # entailment class (last)

            # Backward: candidate entails source
            bwd_enc = self._tokenizer(
                batch_cand, batch_src,
                return_tensors="pt", padding=True,
                truncation=True, max_length=512,
            ).to(self.device)

            with torch.no_grad():
                bwd_out = self._model(**bwd_enc)
                bwd_probs = F.softmax(bwd_out.logits, dim=-1)
                bwd_ent = bwd_probs[:, -1]

            combined = ((fwd_ent + bwd_ent) / 2).cpu().tolist()
            scores.extend(combined)

        return np.array(scores)
