"""
Task-specific evaluation metrics aligned with BartScore paper.

Three evaluation paradigms:
  - WMT: Concordance-based Kendall τ (preference pairs)
  - SUM: Per-document Spearman/Kendall averaged across documents, with
         special cases for Rank19 (accuracy) and QAGS (Pearson)
  - D2T: Document-level Spearman/Kendall across all documents
"""

import logging
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
from scipy.stats import spearmanr, kendalltau, pearsonr

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# WMT: Concordance-based Kendall τ
# ---------------------------------------------------------------------------

def wmt_kendall_tau(better_scores: List[float], worse_scores: List[float]) -> float:
    """Compute WMT-19 official concordance-based Kendall τ.

    For each segment, checks whether the metric assigns a higher score
    to the human-preferred (better) system output. No numeric human score
    is needed — only the preference label.

    Args:
        better_scores: metric scores for the human-preferred system outputs
        worse_scores: metric scores for the human-dispreferred system outputs

    Returns:
        τ = (concordant - discordant) / (concordant + discordant)
    """
    assert len(better_scores) == len(worse_scores)
    conc, disc = 0, 0
    for b, w in zip(better_scores, worse_scores):
        if b > w:
            conc += 1
        else:
            disc += 1
    total = conc + disc
    if total == 0:
        return 0.0
    return (conc - disc) / total


def evaluate_wmt(
    metric_scores_better: List[float],
    metric_scores_worse: List[float],
) -> Dict[str, float]:
    """Full WMT evaluation: returns Kendall τ and accuracy."""
    tau = wmt_kendall_tau(metric_scores_better, metric_scores_worse)
    acc = sum(
        1 for b, w in zip(metric_scores_better, metric_scores_worse) if b > w
    ) / len(metric_scores_better)
    return {"kendall_tau": tau, "accuracy": acc}


# ---------------------------------------------------------------------------
# SUM: Per-document correlation (standard), accuracy (Rank19), Pearson (QAGS)
# ---------------------------------------------------------------------------

def sum_document_correlation(
    data: dict,
    metric_name: str,
    human_metric: str,
) -> Dict[str, float]:
    """SUM-style per-document Spearman/Kendall, averaged across documents.

    For each document, compute correlation between the metric scores
    and human scores across all systems for that document, then average.

    Args:
        data: the benchmark data dict {doc_id: {sys_summs: {sys: {scores: ...}}}}
        metric_name: key for the automatic metric in scores dict
        human_metric: key for the human metric in scores dict

    Returns:
        dict with 'spearman' and 'kendall_tau' (averaged across documents)
    """
    correlations = []
    for doc_id in data:
        sys_summs = data[doc_id]["sys_summs"]
        pred_scores = []
        human_scores = []
        for sys_name in sys_summs:
            scores = sys_summs[sys_name]["scores"]
            if metric_name not in scores or human_metric not in scores:
                continue
            pred_scores.append(float(scores[metric_name]))
            human_scores.append(float(scores[human_metric]))

        if len(set(pred_scores)) <= 1 or len(set(human_scores)) <= 1:
            continue
        if len(pred_scores) < 2:
            continue

        sp = spearmanr(human_scores, pred_scores)[0]
        kt = kendalltau(human_scores, pred_scores)[0]
        if not np.isnan(sp) and not np.isnan(kt):
            correlations.append([sp, kt])

    if not correlations:
        return {"spearman": 0.0, "kendall_tau": 0.0, "n_docs": 0}

    corr_mat = np.array(correlations)
    return {
        "spearman": float(np.mean(corr_mat[:, 0])),
        "kendall_tau": float(np.mean(corr_mat[:, 1])),
        "n_docs": len(correlations),
    }


def sum_fact_pearson(
    data: dict,
    metric_name: str,
    human_metric: str = "fact",
) -> Dict[str, float]:
    """QAGS-style factuality evaluation using Pearson correlation.

    Each document has a single system output with a factuality score.

    Args:
        data: the benchmark data dict
        metric_name: key for the automatic metric
        human_metric: key for the human metric (default: 'fact')
    """
    human_scores = []
    metric_scores = []
    for doc_id in data:
        sys_summs = data[doc_id]["sys_summs"]
        for sys_name in sys_summs:
            scores = sys_summs[sys_name]["scores"]
            if metric_name not in scores or human_metric not in scores:
                continue
            human_scores.append(float(scores[human_metric]))
            metric_scores.append(float(scores[metric_name]))

    if len(human_scores) < 3:
        return {"pearson": 0.0, "n_samples": 0}

    r, p = pearsonr(human_scores, metric_scores)
    return {"pearson": float(r), "p_value": float(p), "n_samples": len(human_scores)}


def sum_fact_accuracy(
    data: dict,
    metric_name: str,
) -> Dict[str, float]:
    """Rank19-style factuality accuracy.

    Each document has two system outputs: 'correct' and 'incorrect'.
    Accuracy = fraction where metric(correct) > metric(incorrect).
    """
    correct_count = 0
    total = 0
    for doc_id in data:
        sys_summs = data[doc_id]["sys_summs"]
        if "correct" not in sys_summs or "incorrect" not in sys_summs:
            continue
        if metric_name not in sys_summs["correct"]["scores"]:
            continue
        if metric_name not in sys_summs["incorrect"]["scores"]:
            continue

        score_correct = float(sys_summs["correct"]["scores"][metric_name])
        score_incorrect = float(sys_summs["incorrect"]["scores"][metric_name])
        if score_correct > score_incorrect:
            correct_count += 1
        total += 1

    if total == 0:
        return {"accuracy": 0.0, "n_samples": 0}
    return {"accuracy": correct_count / total, "n_samples": total}


def evaluate_sum(
    data: dict,
    metric_name: str,
    human_metric: str,
    eval_type: str,
) -> Dict[str, float]:
    """Dispatch to the correct SUM evaluation method."""
    if eval_type == "document_correlation":
        return sum_document_correlation(data, metric_name, human_metric)
    elif eval_type == "fact_pearson":
        return sum_fact_pearson(data, metric_name, human_metric)
    elif eval_type == "accuracy":
        return sum_fact_accuracy(data, metric_name)
    else:
        raise ValueError(f"Unknown SUM eval_type: {eval_type}")


# ---------------------------------------------------------------------------
# D2T: Document-level correlation
# ---------------------------------------------------------------------------

def d2t_document_correlation(
    data: dict,
    metric_name: str,
    human_metric: str,
) -> Dict[str, float]:
    """D2T-style document-level Spearman/Kendall across all documents.

    Unlike SUM (which averages per-document correlations), D2T computes
    correlation across all documents directly.

    Args:
        data: the benchmark data dict {doc_id: {scores: {...}}}
        metric_name: key for the automatic metric in scores dict
        human_metric: key for the human metric in scores dict
    """
    human_scores = []
    metric_scores = []
    for doc_id in data:
        scores = data[doc_id]["scores"]
        if metric_name not in scores or human_metric not in scores:
            continue
        human_scores.append(float(scores[human_metric]))
        metric_scores.append(float(scores[metric_name]))

    if len(human_scores) < 3:
        return {"spearman": 0.0, "kendall_tau": 0.0, "n_samples": 0}

    sp = spearmanr(human_scores, metric_scores)[0]
    kt = kendalltau(human_scores, metric_scores)[0]
    return {
        "spearman": float(sp) if not np.isnan(sp) else 0.0,
        "kendall_tau": float(kt) if not np.isnan(kt) else 0.0,
        "n_samples": len(human_scores),
    }


def evaluate_d2t(
    data: dict,
    metric_name: str,
    human_metric: str,
) -> Dict[str, float]:
    """Full D2T evaluation."""
    return d2t_document_correlation(data, metric_name, human_metric)


# ---------------------------------------------------------------------------
# Bootstrap significance test (task-agnostic)
# ---------------------------------------------------------------------------

def bootstrap_significance_test(
    data: dict,
    m1: str,
    m2: str,
    human_metric: str,
    task: str,
    eval_type: str = "document_correlation",
    n_resamples: int = 1000,
    seed: int = 666,
) -> int:
    """Bootstrap test: is m1 significantly better than m2?

    Returns:
        1 if m1 significantly better, -1 if m2 significantly better, 0 otherwise
    """
    import random
    random.seed(seed)
    doc_ids = list(data.keys())
    better_count = 0

    for _ in range(n_resamples):
        random.shuffle(doc_ids)
        sub_ids = doc_ids[: int(0.8 * len(doc_ids))]

        if task == "WMT":
            m1_better = [float(data[d]["better"]["scores"].get(m1, 0)) for d in sub_ids]
            m1_worse = [float(data[d]["worse"]["scores"].get(m1, 0)) for d in sub_ids]
            m2_better = [float(data[d]["better"]["scores"].get(m2, 0)) for d in sub_ids]
            m2_worse = [float(data[d]["worse"]["scores"].get(m2, 0)) for d in sub_ids]
            m1_ktau = wmt_kendall_tau(m1_better, m1_worse)
            m2_ktau = wmt_kendall_tau(m2_better, m2_worse)
            if m1_ktau > m2_ktau:
                better_count += 1

        elif task == "SUM" and eval_type == "document_correlation":
            sub_data = {d: data[d] for d in sub_ids}
            r1 = sum_document_correlation(sub_data, m1, human_metric)
            r2 = sum_document_correlation(sub_data, m2, human_metric)
            if r1["spearman"] > r2["spearman"]:
                better_count += 1

        elif task == "D2T":
            sub_data = {d: data[d] for d in sub_ids}
            r1 = d2t_document_correlation(sub_data, m1, human_metric)
            r2 = d2t_document_correlation(sub_data, m2, human_metric)
            if r1["spearman"] > r2["spearman"]:
                better_count += 1

    if better_count > 950:
        return 1
    elif better_count < 50:
        return -1
    return 0
