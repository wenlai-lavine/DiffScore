"""
Multi-Timestep Quality Profile (Section 2.2.2).

Decomposes the single ELBO scalar into per-timestep quality scores,
enabling multi-granularity evaluation:
  - Low t (rich context)  -> local fluency / grammar
  - High t (sparse context) -> global coherence / topic consistency
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
from scipy.optimize import minimize
from .diffscore import DiffScoreResult
import logging

logger = logging.getLogger(__name__)


class QualityProfiler:
    """Analyze and aggregate multi-timestep quality profiles."""

    def __init__(self, T: int = 10):
        self.T = T
        self.timesteps = [k / T for k in range(1, T + 1)]

    def extract_profile(self, result: DiffScoreResult) -> np.ndarray:
        """Extract quality profile vector from a DiffScoreResult."""
        return np.array([result.profile.get(t, 0.0) for t in self.timesteps])

    def extract_profiles_batch(
        self, results: List[DiffScoreResult]
    ) -> np.ndarray:
        """Extract profiles for a batch. Returns (N, T) array."""
        return np.array([self.extract_profile(r) for r in results])

    def aggregate_uniform(self, profile: np.ndarray) -> float:
        """Uniform weight aggregation (equivalent to standard ELBO)."""
        return float(np.mean(profile))

    def aggregate_weighted(
        self, profile: np.ndarray, weights: np.ndarray
    ) -> float:
        """Weighted aggregation with learned or manual weights."""
        w = weights / weights.sum()
        return float(np.dot(profile, w))

    def learn_weights(
        self,
        profiles: np.ndarray,
        human_scores: np.ndarray,
        method: str = "spearman",
        n_restarts: int = 10,
        cv_folds: int = 5,
    ) -> np.ndarray:
        """Learn optimal aggregation weights from annotated data.

        Uses multiple random restarts and cross-validation to avoid
        degenerate uniform solutions.

        Args:
            profiles: (N, T) quality profiles
            human_scores: (N,) human annotation scores
            method: 'spearman' or 'pearson' correlation to optimize
            n_restarts: number of random restarts for optimization
            cv_folds: number of cross-validation folds (0 to disable)

        Returns:
            optimal_weights: (T,) weight vector
        """
        from scipy.stats import spearmanr, pearsonr

        corr_fn = spearmanr if method == "spearman" else pearsonr

        def neg_correlation(w, profs, scores):
            w_normed = np.exp(w) / np.exp(w).sum()
            aggregated = profs @ w_normed
            corr, _ = corr_fn(aggregated, scores)
            return -corr if not np.isnan(corr) else 0.0

        def optimize_single(profs, scores, seed=None):
            rng = np.random.RandomState(seed)
            best_result = None
            best_loss = float("inf")

            for _ in range(n_restarts):
                w0 = rng.randn(self.T) * 0.5
                result = minimize(
                    neg_correlation, w0, args=(profs, scores),
                    method="Nelder-Mead",
                    options={"maxiter": 10000, "xatol": 1e-8, "fatol": 1e-8},
                )
                if result.fun < best_loss:
                    best_loss = result.fun
                    best_result = result

            return best_result

        if cv_folds > 1 and len(profiles) >= cv_folds * 5:
            from sklearn.model_selection import KFold
            kf = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
            fold_weights = []
            for train_idx, _ in kf.split(profiles):
                result = optimize_single(
                    profiles[train_idx], human_scores[train_idx], seed=42,
                )
                w = np.exp(result.x) / np.exp(result.x).sum()
                fold_weights.append(w)
            optimal_w = np.mean(fold_weights, axis=0)
            optimal_w = optimal_w / optimal_w.sum()
        else:
            result = optimize_single(profiles, human_scores, seed=42)
            optimal_w = np.exp(result.x) / np.exp(result.x).sum()

        final_corr = corr_fn(profiles @ optimal_w, human_scores)[0]
        logger.info(
            f"Learned weights: {optimal_w.round(4).tolist()}, "
            f"final corr: {final_corr:.4f}"
        )
        return optimal_w

    def per_timestep_correlation(
        self,
        profiles: np.ndarray,
        human_scores: np.ndarray,
        method: str = "spearman",
    ) -> Dict[float, float]:
        """Compute correlation at each timestep individually.

        Used for Experiment 2: verifying that different quality dimensions
        peak at different timesteps.

        Returns:
            Dict mapping timestep -> correlation value
        """
        from scipy.stats import spearmanr, kendalltau, pearsonr

        corr_fn = {"spearman": spearmanr, "kendall": kendalltau, "pearson": pearsonr}
        fn = corr_fn[method]

        correlations = {}
        for k, t in enumerate(self.timesteps):
            scores_at_t = profiles[:, k]
            corr, pval = fn(scores_at_t, human_scores)
            correlations[t] = corr if not np.isnan(corr) else 0.0

        return correlations

    def dimension_timestep_matrix(
        self,
        profiles: np.ndarray,
        human_scores_dict: Dict[str, np.ndarray],
        method: str = "spearman",
    ) -> Dict[str, Dict[float, float]]:
        """Compute dimension x timestep correlation matrix.

        Used for the heatmap in Experiment 2.

        Args:
            profiles: (N, T) array
            human_scores_dict: {dimension_name: (N,) scores}

        Returns:
            {dimension_name: {timestep: correlation}}
        """
        matrix = {}
        for dim_name, scores in human_scores_dict.items():
            matrix[dim_name] = self.per_timestep_correlation(
                profiles, scores, method
            )
        return matrix
