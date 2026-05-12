"""
Correlation metrics for meta-evaluation.

Segment-level: Kendall's tau, Spearman's rho
System-level: Pearson's r, Spearman's rho
"""

import numpy as np
from scipy.stats import kendalltau, spearmanr, pearsonr
from typing import Dict, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


def compute_correlations(
    predictions: np.ndarray,
    human_scores: np.ndarray,
    level: str = "segment",
) -> Dict[str, Tuple[float, float]]:
    """Compute correlation metrics between predicted and human scores.

    Args:
        predictions: (N,) model predictions
        human_scores: (N,) human annotations
        level: 'segment' or 'system'

    Returns:
        dict mapping metric_name -> (correlation, p_value)
    """
    mask = ~(np.isnan(predictions) | np.isnan(human_scores))
    pred = predictions[mask]
    human = human_scores[mask]

    if len(pred) < 3:
        logger.warning("Too few valid samples for correlation computation")
        return {}

    results = {}

    if level == "segment":
        tau, p_tau = kendalltau(pred, human)
        rho, p_rho = spearmanr(pred, human)
        results["kendall_tau"] = (tau, p_tau)
        results["spearman_rho"] = (rho, p_rho)
    elif level == "system":
        r, p_r = pearsonr(pred, human)
        rho, p_rho = spearmanr(pred, human)
        results["pearson_r"] = (r, p_r)
        results["spearman_rho"] = (rho, p_rho)
    else:
        tau, p_tau = kendalltau(pred, human)
        rho_s, p_rho_s = spearmanr(pred, human)
        r, p_r = pearsonr(pred, human)
        results["kendall_tau"] = (tau, p_tau)
        results["spearman_rho"] = (rho_s, p_rho_s)
        results["pearson_r"] = (r, p_r)

    return results


def compute_system_level(
    predictions: np.ndarray,
    human_scores: np.ndarray,
    system_ids: np.ndarray,
) -> Dict[str, Tuple[float, float]]:
    """Aggregate to system-level and compute correlations."""
    unique_systems = np.unique(system_ids)
    sys_pred = []
    sys_human = []

    for sid in unique_systems:
        mask = system_ids == sid
        sys_pred.append(predictions[mask].mean())
        sys_human.append(human_scores[mask].mean())

    return compute_correlations(
        np.array(sys_pred), np.array(sys_human), level="system"
    )


def compute_all_levels(
    predictions: np.ndarray,
    human_scores: np.ndarray,
    system_ids: Optional[np.ndarray] = None,
) -> Dict[str, Dict[str, Tuple[float, float]]]:
    """Compute both segment-level and system-level correlations.

    Returns:
        {"segment": {...}, "system": {...}} where system is only present
        when system_ids are provided and there are >= 3 unique systems.
    """
    result = {"segment": compute_correlations(predictions, human_scores, "segment")}
    if system_ids is not None:
        unique = np.unique(system_ids)
        if len(unique) >= 3:
            result["system"] = compute_system_level(predictions, human_scores, system_ids)
    return result
