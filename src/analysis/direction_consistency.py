"""
Bidirectional Reasoning Consistency Analysis (Section 2.3.3 / Experiment 4).

Measures whether forward and reverse formulations of the same fact
receive consistent scores under different evaluation models.

DirConsist = 1 - |Score(forward) - Score(reverse)| / (|Score(forward)| + |Score(reverse)|)
"""

import numpy as np
from typing import List, Tuple, Dict
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class ConsistencyResult:
    """Results of direction consistency analysis."""
    mean_consistency: float
    std_consistency: float
    per_pair_consistency: np.ndarray
    forward_scores: np.ndarray
    reverse_scores: np.ndarray


class DirectionConsistencyAnalyzer:
    """Analyze bidirectional reasoning consistency."""

    @staticmethod
    def compute_consistency(
        forward_scores: np.ndarray,
        reverse_scores: np.ndarray,
    ) -> ConsistencyResult:
        """Compute direction consistency ratio for score pairs.

        DirConsist = 1 - |S(fwd) - S(rev)| / (|S(fwd)| + |S(rev)|)

        Values close to 1.0 indicate high consistency.
        """
        abs_diff = np.abs(forward_scores - reverse_scores)
        abs_sum = np.abs(forward_scores) + np.abs(reverse_scores)

        # Avoid division by zero
        safe_sum = np.maximum(abs_sum, 1e-10)
        per_pair = 1.0 - abs_diff / safe_sum

        return ConsistencyResult(
            mean_consistency=float(np.mean(per_pair)),
            std_consistency=float(np.std(per_pair)),
            per_pair_consistency=per_pair,
            forward_scores=forward_scores,
            reverse_scores=reverse_scores,
        )

    def evaluate_diffscore(
        self,
        scorer,
        pairs: List[Tuple[str, str]],
    ) -> ConsistencyResult:
        """Evaluate DiffScore's direction consistency on statement pairs.

        Args:
            scorer: DiffScorer instance
            pairs: list of (forward_statement, reverse_statement)
        """
        forward_scores = []
        reverse_scores = []

        for fwd, rev in pairs:
            fwd_result = scorer.score_marginal(fwd)
            rev_result = scorer.score_marginal(rev)
            forward_scores.append(fwd_result.scalar)
            reverse_scores.append(rev_result.scalar)

        return self.compute_consistency(
            np.array(forward_scores), np.array(reverse_scores)
        )

    def evaluate_ar_model(
        self,
        model,
        tokenizer,
        pairs: List[Tuple[str, str]],
        device: str = "cuda",
    ) -> ConsistencyResult:
        """Evaluate AR model's direction consistency."""
        import torch
        import torch.nn.functional as F

        forward_scores = []
        reverse_scores = []

        for fwd, rev in pairs:
            for text, score_list in [(fwd, forward_scores), (rev, reverse_scores)]:
                enc = tokenizer.encode(text, add_special_tokens=True)
                input_ids = torch.tensor([enc], dtype=torch.long).to(device)

                with torch.no_grad():
                    outputs = model(input_ids=input_ids)
                    logits = outputs.logits

                log_probs = F.log_softmax(logits, dim=-1)
                token_lps = []
                for pos in range(1, len(enc)):
                    lp = log_probs[0, pos - 1, enc[pos]].item()
                    token_lps.append(lp)

                score_list.append(np.mean(token_lps) if token_lps else 0.0)

        return self.compute_consistency(
            np.array(forward_scores), np.array(reverse_scores)
        )

    @staticmethod
    def compare_methods(
        results: Dict[str, ConsistencyResult]
    ) -> Dict[str, Dict[str, float]]:
        """Compare consistency across methods.

        Returns summary table-ready dict.
        """
        summary = {}
        for name, res in results.items():
            summary[name] = {
                "mean_consistency": res.mean_consistency,
                "std_consistency": res.std_consistency,
                "score_correlation": float(
                    np.corrcoef(res.forward_scores, res.reverse_scores)[0, 1]
                ),
            }
        return summary
