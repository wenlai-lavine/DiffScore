"""
Bootstrap confidence intervals for correlation estimates.

95% Bootstrap CI following MT evaluation best practices.
"""

import numpy as np
from scipy.stats import spearmanr, kendalltau, pearsonr
from typing import Tuple, Optional, Callable
import logging

logger = logging.getLogger(__name__)


def bootstrap_confidence_interval(
    predictions: np.ndarray,
    human_scores: np.ndarray,
    metric: str = "spearman",
    n_resamples: int = 1000,
    confidence_level: float = 0.95,
    seed: int = 42,
) -> Tuple[float, float, float]:
    """Compute bootstrap confidence interval for a correlation metric.

    Args:
        predictions: (N,) predicted scores
        human_scores: (N,) human annotation scores
        metric: 'spearman', 'kendall', or 'pearson'
        n_resamples: number of bootstrap resamples
        confidence_level: CI level (default 0.95)

    Returns:
        (point_estimate, ci_lower, ci_upper)
    """
    corr_fn = {
        "spearman": lambda x, y: spearmanr(x, y)[0],
        "kendall": lambda x, y: kendalltau(x, y)[0],
        "pearson": lambda x, y: pearsonr(x, y)[0],
    }[metric]

    point = corr_fn(predictions, human_scores)

    rng = np.random.RandomState(seed)
    n = len(predictions)
    boot_corrs = []

    for _ in range(n_resamples):
        indices = rng.randint(0, n, size=n)
        boot_pred = predictions[indices]
        boot_human = human_scores[indices]
        try:
            c = corr_fn(boot_pred, boot_human)
            if not np.isnan(c):
                boot_corrs.append(c)
        except Exception:
            continue

    boot_corrs = np.array(boot_corrs)
    alpha = 1 - confidence_level
    ci_lower = np.percentile(boot_corrs, 100 * alpha / 2)
    ci_upper = np.percentile(boot_corrs, 100 * (1 - alpha / 2))

    return point, ci_lower, ci_upper


def paired_bootstrap_test(
    predictions_a: np.ndarray,
    predictions_b: np.ndarray,
    human_scores: np.ndarray,
    metric: str = "spearman",
    n_resamples: int = 1000,
    seed: int = 42,
) -> Tuple[float, float]:
    """Paired bootstrap significance test between two metrics.

    Tests H0: corr(A, human) = corr(B, human).

    Returns:
        (delta, p_value): difference in correlation and p-value
    """
    corr_fn = {
        "spearman": lambda x, y: spearmanr(x, y)[0],
        "kendall": lambda x, y: kendalltau(x, y)[0],
        "pearson": lambda x, y: pearsonr(x, y)[0],
    }[metric]

    point_a = corr_fn(predictions_a, human_scores)
    point_b = corr_fn(predictions_b, human_scores)
    observed_delta = point_a - point_b

    rng = np.random.RandomState(seed)
    n = len(human_scores)
    count_more_extreme = 0

    for _ in range(n_resamples):
        indices = rng.randint(0, n, size=n)
        ba = predictions_a[indices]
        bb = predictions_b[indices]
        bh = human_scores[indices]
        try:
            ca = corr_fn(ba, bh)
            cb = corr_fn(bb, bh)
            delta = ca - cb
            if delta <= 0:
                count_more_extreme += 1
        except Exception:
            continue

    p_value = count_more_extreme / n_resamples
    return observed_delta, p_value
