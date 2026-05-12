"""
Embedding-based metrics: BERTScore, MoverScore.
"""

import numpy as np
from typing import List, Dict, Optional
import logging

logger = logging.getLogger(__name__)


class BERTScoreWrapper:
    """Wrapper around the bert-score library."""

    def __init__(
        self,
        model_type: str = "microsoft/deberta-xlarge-mnli",
        device: str = "cuda",
        batch_size: int = 32,
    ):
        self.model_type = model_type
        self.device = device
        self.batch_size = batch_size

    def score(
        self,
        candidates: List[str],
        references: List[str],
    ) -> Dict[str, np.ndarray]:
        """Compute BERTScore (P, R, F1)."""
        from bert_score import score as bert_score_fn

        P, R, F1 = bert_score_fn(
            candidates, references,
            model_type=self.model_type,
            device=self.device,
            batch_size=self.batch_size,
            verbose=False,
        )

        return {
            "bertscore_p": P.numpy(),
            "bertscore_r": R.numpy(),
            "bertscore_f1": F1.numpy(),
        }
