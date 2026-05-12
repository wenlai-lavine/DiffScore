"""
PMI (Pointwise Mutual Information) Fluency-Relevance Decoupling (Section 2.2.3).

DiffScore_PMI(c, s) = DiffScore_cond(c|s) - DiffScore_mar(c)

Decomposes overall quality into:
  - Fluency: DiffScore_mar(c)
  - Relevance: DiffScore_PMI(c, s)  [information gain from source]
"""

import numpy as np
from typing import Dict, List, Optional
from dataclasses import dataclass
from .diffscore import DiffScorer, DiffScoreResult
import logging

logger = logging.getLogger(__name__)


@dataclass
class PMIResult:
    """Container for PMI decomposition results."""
    overall: float           # DiffScore_cond(c|s)
    fluency: float           # DiffScore_mar(c)
    relevance: float         # PMI = overall - fluency
    overall_profile: Dict[float, float]
    fluency_profile: Dict[float, float]
    pmi_profile: Dict[float, float]  # per-timestep PMI


class PMIScorer:
    """Compute PMI-based fluency-relevance decomposition."""

    def __init__(self, scorer: DiffScorer):
        self.scorer = scorer

    def score(
        self,
        source: str,
        candidate: str,
        prompt_template: Dict[str, str],
    ) -> PMIResult:
        """Compute full PMI decomposition for a source-candidate pair."""
        cond_result = self.scorer.score_conditional(
            source, candidate, prompt_template, return_profile=True
        )
        mar_result = self.scorer.score_marginal(
            candidate, return_profile=True
        )

        pmi_scalar = cond_result.scalar - mar_result.scalar

        pmi_profile = {}
        for t in cond_result.profile:
            cond_t = cond_result.profile.get(t, 0.0)
            mar_t = mar_result.profile.get(t, 0.0)
            pmi_profile[t] = cond_t - mar_t

        return PMIResult(
            overall=cond_result.scalar,
            fluency=mar_result.scalar,
            relevance=pmi_scalar,
            overall_profile=cond_result.raw_profile,
            fluency_profile=mar_result.raw_profile,
            pmi_profile=pmi_profile,
        )

    def score_batch(
        self,
        sources: List[str],
        candidates: List[str],
        prompt_template: Dict[str, str],
        show_progress: bool = True,
    ) -> List[PMIResult]:
        """Compute PMI decomposition for a batch of pairs."""
        from tqdm import tqdm

        results = []
        pairs = list(zip(sources, candidates))
        iterator = tqdm(pairs, desc="DiffScore-PMI") if show_progress else pairs

        for src, cand in iterator:
            results.append(self.score(src, cand, prompt_template))

        return results

    @staticmethod
    def extract_scores(
        results: List[PMIResult], component: str = "relevance"
    ) -> np.ndarray:
        """Extract a specific component from PMI results.

        Args:
            component: 'overall', 'fluency', or 'relevance'
        """
        return np.array([getattr(r, component) for r in results])
