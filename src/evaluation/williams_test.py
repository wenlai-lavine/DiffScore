"""
Williams significance test for comparing dependent correlations.

Tests whether two correlations r(A, H) and r(B, H) are significantly different,
accounting for the correlation between A and B.
"""

import numpy as np
from scipy.stats import t as t_dist, pearsonr, spearmanr
from typing import Tuple
import logging

logger = logging.getLogger(__name__)


def williams_test(
    predictions_a: np.ndarray,
    predictions_b: np.ndarray,
    human_scores: np.ndarray,
) -> Tuple[float, float]:
    """Williams test for dependent correlation comparison.

    Tests H0: rho(A, H) = rho(B, H) where A and B are correlated.

    Based on Steiger (1980) and used in MT evaluation (Graham & Baldwin, 2014).

    Args:
        predictions_a: (N,) scores from metric A
        predictions_b: (N,) scores from metric B
        human_scores:  (N,) human annotation scores

    Returns:
        (t_statistic, p_value)
    """
    n = len(human_scores)

    r_ah = pearsonr(predictions_a, human_scores)[0]
    r_bh = pearsonr(predictions_b, human_scores)[0]
    r_ab = pearsonr(predictions_a, predictions_b)[0]

    # Williams (1959) / Steiger (1980) formula
    r_mean_sq = (r_ah**2 + r_bh**2) / 2.0
    det = 1 - r_ah**2 - r_bh**2 - r_ab**2 + 2 * r_ah * r_bh * r_ab

    numerator = (r_ah - r_bh) * np.sqrt(
        (n - 1) * (1 + r_ab)
    )
    denominator = np.sqrt(
        2 * ((n - 1) / (n - 3)) * det + r_mean_sq * (1 - r_ab) ** 3
    )

    if denominator < 1e-10:
        logger.warning("Williams test denominator near zero; correlations may be identical")
        return 0.0, 1.0

    t_stat = numerator / denominator

    df = n - 3
    p_value = 2 * (1 - t_dist.cdf(abs(t_stat), df))

    return float(t_stat), float(p_value)


def williams_test_spearman(
    predictions_a: np.ndarray,
    predictions_b: np.ndarray,
    human_scores: np.ndarray,
) -> Tuple[float, float]:
    """Williams test using rank correlations (Spearman).

    Converts to ranks first, then applies the Pearson-based Williams test.
    """
    from scipy.stats import rankdata

    rank_a = rankdata(predictions_a)
    rank_b = rankdata(predictions_b)
    rank_h = rankdata(human_scores)

    return williams_test(rank_a, rank_b, rank_h)
