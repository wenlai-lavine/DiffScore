"""
Experiment 1: Main Meta-Evaluation (BartScore-aligned benchmarks).

Evaluates DiffScore variants and baselines across three task categories
aligned with the BartScore paper (Yuan et al., 2021):

  - WMT (Machine Translation): 7 language pairs, preference-based Kendall τ
  - SUM (Summarization): 6 datasets with varying human metrics
  - D2T (Data-to-Text): 3 datasets with informativeness/naturalness/quality

Reports task-appropriate correlation metrics with bootstrap significance tests.
"""

import os
import sys
import json
import logging
import argparse
import time
from collections import defaultdict

import numpy as np
import torch
from tabulate import tabulate
from typing import List, Dict, Any, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model import MDLMWrapper
from src.scoring import DiffScorer
from src.data.bartscore_benchmark import (
    WMTBenchmark, SUMBenchmark, D2TBenchmark,
    BENCHMARK_REGISTRY, load_benchmark,
    get_benchmarks_by_task, get_all_benchmark_names,
)
from src.evaluation.bartscore_eval import (
    evaluate_wmt, evaluate_sum, evaluate_d2t,
    wmt_kendall_tau, sum_document_correlation,
    d2t_document_correlation,
)
from src.baselines.bartscore import BARTScorer as BARTScorerMultiConfig
from src.baselines.bartscore_official import BARTScorer as BARTScorerOfficial
from src.baselines.traditional import BLEUScorer, ROUGEScorer, METEORScorer
from src.baselines.embedding import BERTScoreWrapper
from src.baselines.moverscore import MoverScorer
from src.baselines.alignscore import AlignScorer
from src.baselines.questeval import QuestEvalScorer
from src.baselines.unieval import UniEvalScorer
from src.baselines.geval import GEvalScorer
from src.baselines.unieval_official import convert_to_json
from src.utils import release_model

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# DiffScore computation per task
# ---------------------------------------------------------------------------

def compute_diffscore_wmt(
    model, benchmark: WMTBenchmark, prompt_template: dict,
    K: int = 20, T: int = 10, batch_size: int = 4,
    scoring_mode: str = "mean_lp", max_samples: Optional[int] = None,
    seed: int = 42,
) -> Dict[str, Dict[str, List[float]]]:
    """Compute DiffScore for WMT preference pairs.

    Returns dict with metric_name -> {'better': [...], 'worse': [...]}.
    """
    _set_seed(seed)

    refs = benchmark.get_refs()
    betters = benchmark.get_betters()
    worses = benchmark.get_worses()

    if max_samples:
        refs = refs[:max_samples]
        betters = betters[:max_samples]
        worses = worses[:max_samples]

    scorer = DiffScorer(model, K=K, T=T, batch_size=batch_size, scoring_mode=scoring_mode)
    results = {}

    # Conditional: score(candidate | ref) — faithfulness to reference
    logger.info("  DiffScore conditional (ref → better)...")
    cond_better = scorer.score_batch(
        sources=refs, candidates=betters, prompt_template=prompt_template,
        configuration="conditional",
    )
    logger.info("  DiffScore conditional (ref → worse)...")
    cond_worse = scorer.score_batch(
        sources=refs, candidates=worses, prompt_template=prompt_template,
        configuration="conditional",
    )
    results["diffscore_cond"] = {
        "better": [r.scalar for r in cond_better],
        "worse": [r.scalar for r in cond_worse],
    }

    # Reverse: score(ref | candidate) — coverage of reference
    logger.info("  DiffScore reverse (better → ref)...")
    rev_better = scorer.score_batch(
        sources=refs, candidates=betters, prompt_template=prompt_template,
        configuration="reverse",
    )
    logger.info("  DiffScore reverse (worse → ref)...")
    rev_worse = scorer.score_batch(
        sources=refs, candidates=worses, prompt_template=prompt_template,
        configuration="reverse",
    )
    results["diffscore_rev"] = {
        "better": [r.scalar for r in rev_better],
        "worse": [r.scalar for r in rev_worse],
    }

    # Marginal: score(candidate) — fluency
    logger.info("  DiffScore marginal (better)...")
    mar_better = scorer.score_batch(texts=betters, configuration="marginal")
    logger.info("  DiffScore marginal (worse)...")
    mar_worse = scorer.score_batch(texts=worses, configuration="marginal")
    results["diffscore_mar"] = {
        "better": [r.scalar for r in mar_better],
        "worse": [r.scalar for r in mar_worse],
    }

    # Derived configurations
    cond_b = np.array(results["diffscore_cond"]["better"])
    cond_w = np.array(results["diffscore_cond"]["worse"])
    rev_b = np.array(results["diffscore_rev"]["better"])
    rev_w = np.array(results["diffscore_rev"]["worse"])
    mar_b = np.array(results["diffscore_mar"]["better"])
    mar_w = np.array(results["diffscore_mar"]["worse"])

    for alpha in [0.5, 0.7]:
        key = f"diffscore_bi_a{int(alpha*10)}"
        results[key] = {
            "better": (alpha * cond_b + (1 - alpha) * rev_b).tolist(),
            "worse": (alpha * cond_w + (1 - alpha) * rev_w).tolist(),
        }

    results["diffscore_pmi"] = {
        "better": (cond_b - mar_b).tolist(),
        "worse": (cond_w - mar_w).tolist(),
    }

    return results


def compute_diffscore_sum(
    model, benchmark: SUMBenchmark, prompt_template: dict,
    K: int = 20, T: int = 10, batch_size: int = 4,
    scoring_mode: str = "mean_lp", max_samples: Optional[int] = None,
    seed: int = 42,
) -> Dict[str, str]:
    """Compute DiffScore for SUM benchmarks.

    Stores scores directly in the benchmark data dict.
    Returns list of metric names that were computed.
    """
    _set_seed(seed)

    scorer = DiffScorer(model, K=K, T=T, batch_size=batch_size, scoring_mode=scoring_mode)

    # Collect all (src, sys_summ) pairs for batched scoring
    all_src, all_cand, all_keys = [], [], []
    sample_count = 0
    for doc_id, sys_name, src, sys_summ in benchmark.iter_all_samples():
        all_src.append(src)
        all_cand.append(sys_summ)
        all_keys.append((doc_id, sys_name))
        sample_count += 1
        if max_samples and sample_count >= max_samples:
            break

    metric_names = []

    # Conditional (src → candidate)
    logger.info(f"  DiffScore conditional: {len(all_cand)} samples...")
    cond_results = scorer.score_batch(
        sources=all_src, candidates=all_cand,
        prompt_template=prompt_template, configuration="conditional",
    )
    for (doc_id, sys_name), result in zip(all_keys, cond_results):
        benchmark.store_metric_score(doc_id, sys_name, "diffscore_cond", result.scalar)
    metric_names.append("diffscore_cond")

    # Marginal (candidate only)
    logger.info(f"  DiffScore marginal: {len(all_cand)} samples...")
    mar_results = scorer.score_batch(texts=all_cand, configuration="marginal")
    for (doc_id, sys_name), result in zip(all_keys, mar_results):
        benchmark.store_metric_score(doc_id, sys_name, "diffscore_mar", result.scalar)
    metric_names.append("diffscore_mar")

    # Reverse (candidate → src)
    logger.info(f"  DiffScore reverse: {len(all_cand)} samples...")
    rev_results = scorer.score_batch(
        sources=all_src, candidates=all_cand,
        prompt_template=prompt_template, configuration="reverse",
    )
    for (doc_id, sys_name), result in zip(all_keys, rev_results):
        benchmark.store_metric_score(doc_id, sys_name, "diffscore_rev", result.scalar)
    metric_names.append("diffscore_rev")

    # Derived: bidirectional & PMI
    for doc_id, sys_name in all_keys:
        scores = benchmark.data[doc_id]["sys_summs"][sys_name]["scores"]
        c = scores["diffscore_cond"]
        r = scores["diffscore_rev"]
        m = scores["diffscore_mar"]
        scores["diffscore_bi_a5"] = 0.5 * c + 0.5 * r
        scores["diffscore_bi_a7"] = 0.7 * c + 0.3 * r
        scores["diffscore_pmi"] = c - m

    metric_names.extend(["diffscore_bi_a5", "diffscore_bi_a7", "diffscore_pmi"])
    return metric_names


def compute_diffscore_d2t(
    model, benchmark: D2TBenchmark, prompt_template: dict,
    K: int = 20, T: int = 10, batch_size: int = 4,
    scoring_mode: str = "mean_lp", max_samples: Optional[int] = None,
    seed: int = 42,
) -> List[str]:
    """Compute DiffScore for D2T benchmarks.

    Stores scores directly in the benchmark data dict.
    Returns list of metric names that were computed.
    """
    _set_seed(seed)

    scorer = DiffScorer(model, K=K, T=T, batch_size=batch_size, scoring_mode=scoring_mode)

    srcs = benchmark.get_src_lines()
    cands = benchmark.get_sys_summs()
    doc_ids = benchmark.doc_ids

    if max_samples:
        srcs = srcs[:max_samples]
        cands = cands[:max_samples]
        doc_ids = doc_ids[:max_samples]

    ref_summs_all = benchmark.get_ref_summs()
    if max_samples:
        ref_summs_all = ref_summs_all[:max_samples]

    metric_names = []

    # Conditional (src → candidate)
    logger.info(f"  DiffScore conditional (src): {len(cands)} samples...")
    cond_results = scorer.score_batch(
        sources=srcs, candidates=cands,
        prompt_template=prompt_template, configuration="conditional",
    )
    for doc_id, result in zip(doc_ids, cond_results):
        benchmark.store_metric_score(doc_id, "diffscore_cond_src", result.scalar)
    metric_names.append("diffscore_cond_src")

    # Conditional (ref → candidate): average over references
    logger.info(f"  DiffScore conditional (ref): {len(cands)} samples...")
    n_refs = len(ref_summs_all[0]) if ref_summs_all else 1
    ref_cond_scores = np.zeros(len(cands))
    for ref_idx in range(min(n_refs, 3)):  # limit to 3 refs for efficiency
        ref_list = [refs[ref_idx] if ref_idx < len(refs) else refs[0] for refs in ref_summs_all]
        ref_results = scorer.score_batch(
            sources=ref_list, candidates=cands,
            prompt_template=prompt_template, configuration="conditional",
        )
        ref_cond_scores += np.array([r.scalar for r in ref_results])
    ref_cond_scores /= min(n_refs, 3)
    for doc_id, score in zip(doc_ids, ref_cond_scores):
        benchmark.store_metric_score(doc_id, "diffscore_cond_ref", float(score))
    metric_names.append("diffscore_cond_ref")

    # Marginal (candidate only)
    logger.info(f"  DiffScore marginal: {len(cands)} samples...")
    mar_results = scorer.score_batch(texts=cands, configuration="marginal")
    for doc_id, result in zip(doc_ids, mar_results):
        benchmark.store_metric_score(doc_id, "diffscore_mar", result.scalar)
    metric_names.append("diffscore_mar")

    # Reverse (candidate → ref): average over references
    logger.info(f"  DiffScore reverse (ref): {len(cands)} samples...")
    rev_scores = np.zeros(len(cands))
    for ref_idx in range(min(n_refs, 3)):
        ref_list = [refs[ref_idx] if ref_idx < len(refs) else refs[0] for refs in ref_summs_all]
        rev_results = scorer.score_batch(
            sources=ref_list, candidates=cands,
            prompt_template=prompt_template, configuration="reverse",
        )
        rev_scores += np.array([r.scalar for r in rev_results])
    rev_scores /= min(n_refs, 3)
    for doc_id, score in zip(doc_ids, rev_scores):
        benchmark.store_metric_score(doc_id, "diffscore_rev_ref", float(score))
    metric_names.append("diffscore_rev_ref")

    # Derived: bidirectional & PMI
    for i, doc_id in enumerate(doc_ids):
        scores = benchmark.data[doc_id]["scores"]
        c_ref = scores.get("diffscore_cond_ref", 0)
        r_ref = scores.get("diffscore_rev_ref", 0)
        m = scores.get("diffscore_mar", 0)
        c_src = scores.get("diffscore_cond_src", 0)
        scores["diffscore_bi_a5"] = 0.5 * c_ref + 0.5 * r_ref
        scores["diffscore_bi_a7"] = 0.7 * c_ref + 0.3 * r_ref
        scores["diffscore_pmi_src"] = c_src - m
        scores["diffscore_pmi_ref"] = c_ref - m

    metric_names.extend([
        "diffscore_bi_a5", "diffscore_bi_a7",
        "diffscore_pmi_src", "diffscore_pmi_ref",
    ])
    return metric_names


# ---------------------------------------------------------------------------
# Baseline computation per task
# ---------------------------------------------------------------------------

def compute_baselines_wmt(
    benchmark: WMTBenchmark, device: str = "cuda",
    max_samples: Optional[int] = None,
) -> Dict[str, Dict[str, List[float]]]:
    """Compute baseline scores for WMT (better/worse for each metric)."""
    refs = benchmark.get_refs()
    betters = benchmark.get_betters()
    worses = benchmark.get_worses()

    if max_samples:
        refs = refs[:max_samples]
        betters = betters[:max_samples]
        worses = worses[:max_samples]

    results = {}

    # BERTScore
    logger.info("  BERTScore...")
    bert = BERTScoreWrapper(device=device)
    bert_better = bert.score(betters, refs)
    bert_worse = bert.score(worses, refs)
    results["bertscore_f1"] = {
        "better": bert_better["bertscore_f1"].tolist(),
        "worse": bert_worse["bertscore_f1"].tolist(),
    }
    release_model(bert)
    del bert

    # BARTScore (ref → hypo and hypo → ref, averaged)
    logger.info("  BARTScore...")
    bart = BARTScorerOfficial(device=device)
    bart_ref_better = np.array(bart.score(refs, betters))
    bart_better_ref = np.array(bart.score(betters, refs))
    bart_ref_worse = np.array(bart.score(refs, worses))
    bart_worse_ref = np.array(bart.score(worses, refs))
    results["bartscore_avg_f"] = {
        "better": (0.5 * (bart_ref_better + bart_better_ref)).tolist(),
        "worse": (0.5 * (bart_ref_worse + bart_worse_ref)).tolist(),
    }
    results["bartscore_ref_hypo"] = {
        "better": bart_ref_better.tolist(),
        "worse": bart_ref_worse.tolist(),
    }
    results["bartscore_hypo_ref"] = {
        "better": bart_better_ref.tolist(),
        "worse": bart_worse_ref.tolist(),
    }
    release_model(bart)
    del bart

    # Traditional: BLEU, METEOR
    logger.info("  BLEU / METEOR...")
    bleu = BLEUScorer()
    results["bleu"] = {
        "better": bleu.score(betters, refs).tolist(),
        "worse": bleu.score(worses, refs).tolist(),
    }
    meteor = METEORScorer()
    results["meteor"] = {
        "better": meteor.score(betters, refs).tolist(),
        "worse": meteor.score(worses, refs).tolist(),
    }

    # MoverScore
    logger.info("  MoverScore...")
    mover = MoverScorer(device=device)
    results["moverscore"] = {
        "better": mover.score(betters, refs).tolist(),
        "worse": mover.score(worses, refs).tolist(),
    }
    release_model(mover)
    del mover

    # AlignScore (source=ref, candidate=hypo)
    logger.info("  AlignScore...")
    align = AlignScorer(device=device)
    results["alignscore"] = {
        "better": align.score(refs, betters).tolist(),
        "worse": align.score(refs, worses).tolist(),
    }
    release_model(align)
    del align

    # QuestEval
    logger.info("  QuestEval...")
    questeval = QuestEvalScorer(task="summarization", device=device)
    results["questeval"] = {
        "better": questeval.score(refs, betters).tolist(),
        "worse": questeval.score(refs, worses).tolist(),
    }
    release_model(questeval)
    del questeval

    # UniEval (translation mode for WMT)
    logger.info("  UniEval...")
    unieval = UniEvalScorer(task="translation", device=device)
    unieval_better = unieval.score(refs, betters, references=refs)
    unieval_worse = unieval.score(refs, worses, references=refs)
    for key in unieval_better:
        results[key] = {
            "better": unieval_better[key].tolist(),
            "worse": unieval_worse[key].tolist(),
        }
    release_model(unieval)
    del unieval

    # G-Eval (LLM-as-Judge)
    logger.info("  G-Eval...")
    geval = GEvalScorer()
    geval_better = geval.score(refs, betters)
    geval_worse = geval.score(refs, worses)
    for key in geval_better:
        results[key] = {
            "better": geval_better[key].tolist(),
            "worse": geval_worse[key].tolist(),
        }

    return results


def compute_baselines_sum(
    benchmark: SUMBenchmark, device: str = "cuda",
    max_samples: Optional[int] = None,
) -> List[str]:
    """Compute baseline scores for SUM, stored in benchmark data dict."""
    metric_names = []
    src_lines = benchmark.get_src_lines()
    ref_lines = benchmark.get_single_ref_lines()

    # BERTScore
    logger.info("  BERTScore...")
    bert = BERTScoreWrapper(device=device)
    for sys_name in benchmark.sys_names:
        sys_lines = benchmark.get_sys_lines(sys_name)
        if max_samples:
            sys_lines = sys_lines[:max_samples]
            ref_sub = ref_lines[:max_samples]
        else:
            ref_sub = ref_lines
        scores = bert.score(sys_lines, ref_sub)
        for i, doc_id in enumerate(benchmark.doc_ids[:len(sys_lines)]):
            benchmark.store_metric_score(doc_id, sys_name, "bertscore_f1", float(scores["bertscore_f1"][i]))
    metric_names.append("bertscore_f1")
    release_model(bert)
    del bert

    # BARTScore
    logger.info("  BARTScore...")
    bart = BARTScorerOfficial(device=device)
    for sys_name in benchmark.sys_names:
        sys_lines = benchmark.get_sys_lines(sys_name)
        if max_samples:
            sys_lines = sys_lines[:max_samples]
            src_sub = src_lines[:max_samples]
            ref_sub = ref_lines[:max_samples]
        else:
            src_sub = src_lines
            ref_sub = ref_lines
        src_hypo = bart.score(src_sub, sys_lines)
        ref_hypo = np.array(bart.score(ref_sub, sys_lines))
        hypo_ref = np.array(bart.score(sys_lines, ref_sub))
        avg_f = (0.5 * (ref_hypo + hypo_ref)).tolist()
        for i, doc_id in enumerate(benchmark.doc_ids[:len(sys_lines)]):
            benchmark.store_metric_score(doc_id, sys_name, "bartscore_src_hypo", float(src_hypo[i]))
            benchmark.store_metric_score(doc_id, sys_name, "bartscore_ref_hypo", float(ref_hypo[i]))
            benchmark.store_metric_score(doc_id, sys_name, "bartscore_hypo_ref", float(hypo_ref[i]))
            benchmark.store_metric_score(doc_id, sys_name, "bartscore_avg_f", float(avg_f[i]))
    metric_names.extend(["bartscore_src_hypo", "bartscore_ref_hypo", "bartscore_hypo_ref", "bartscore_avg_f"])
    release_model(bart)
    del bart

    # ROUGE
    logger.info("  ROUGE...")
    rouge = ROUGEScorer()
    for sys_name in benchmark.sys_names:
        sys_lines = benchmark.get_sys_lines(sys_name)
        if max_samples:
            sys_lines = sys_lines[:max_samples]
            ref_sub = ref_lines[:max_samples]
        else:
            ref_sub = ref_lines
        rouge_scores = rouge.score(sys_lines, ref_sub)
        for i, doc_id in enumerate(benchmark.doc_ids[:len(sys_lines)]):
            for rk, rv in rouge_scores.items():
                benchmark.store_metric_score(doc_id, sys_name, rk, float(rv[i]))
    metric_names.extend(list(rouge_scores.keys()))

    # MoverScore
    logger.info("  MoverScore...")
    mover = MoverScorer(device=device)
    for sys_name in benchmark.sys_names:
        sys_lines = benchmark.get_sys_lines(sys_name)
        if max_samples:
            sys_lines = sys_lines[:max_samples]
            ref_sub = ref_lines[:max_samples]
        else:
            ref_sub = ref_lines
        scores = mover.score(sys_lines, ref_sub)
        for i, doc_id in enumerate(benchmark.doc_ids[:len(sys_lines)]):
            benchmark.store_metric_score(doc_id, sys_name, "moverscore", float(scores[i]))
    metric_names.append("moverscore")
    release_model(mover)
    del mover

    # AlignScore
    logger.info("  AlignScore...")
    align = AlignScorer(device=device)
    for sys_name in benchmark.sys_names:
        sys_lines = benchmark.get_sys_lines(sys_name)
        if max_samples:
            sys_lines = sys_lines[:max_samples]
            src_sub = src_lines[:max_samples]
        else:
            src_sub = src_lines
        scores = align.score(src_sub, sys_lines)
        for i, doc_id in enumerate(benchmark.doc_ids[:len(sys_lines)]):
            benchmark.store_metric_score(doc_id, sys_name, "alignscore", float(scores[i]))
    metric_names.append("alignscore")
    release_model(align)
    del align

    # QuestEval
    logger.info("  QuestEval...")
    questeval = QuestEvalScorer(task="summarization", device=device)
    for sys_name in benchmark.sys_names:
        sys_lines = benchmark.get_sys_lines(sys_name)
        if max_samples:
            sys_lines = sys_lines[:max_samples]
            src_sub = src_lines[:max_samples]
        else:
            src_sub = src_lines
        scores = questeval.score(src_sub, sys_lines)
        for i, doc_id in enumerate(benchmark.doc_ids[:len(sys_lines)]):
            benchmark.store_metric_score(doc_id, sys_name, "questeval", float(scores[i]))
    metric_names.append("questeval")
    release_model(questeval)
    del questeval

    # UniEval (summarization, per-dimension + overall)
    logger.info("  UniEval...")
    unieval = UniEvalScorer(task="summarization", device=device)
    for sys_name in benchmark.sys_names:
        sys_lines = benchmark.get_sys_lines(sys_name)
        if max_samples:
            sys_lines = sys_lines[:max_samples]
            src_sub = src_lines[:max_samples]
            ref_sub = ref_lines[:max_samples]
        else:
            src_sub = src_lines
            ref_sub = ref_lines
        unieval_scores = unieval.score(src_sub, sys_lines, references=ref_sub)
        for dim_key, dim_vals in unieval_scores.items():
            for i, doc_id in enumerate(benchmark.doc_ids[:len(sys_lines)]):
                benchmark.store_metric_score(doc_id, sys_name, dim_key, float(dim_vals[i]))
            if dim_key not in metric_names:
                metric_names.append(dim_key)
    release_model(unieval)
    del unieval

    # G-Eval (LLM-as-Judge)
    logger.info("  G-Eval...")
    geval = GEvalScorer()
    for sys_name in benchmark.sys_names:
        sys_lines = benchmark.get_sys_lines(sys_name)
        if max_samples:
            sys_lines = sys_lines[:max_samples]
            src_sub = src_lines[:max_samples]
        else:
            src_sub = src_lines
        geval_scores = geval.score(src_sub, sys_lines)
        for dim_key, dim_vals in geval_scores.items():
            for i, doc_id in enumerate(benchmark.doc_ids[:len(sys_lines)]):
                benchmark.store_metric_score(doc_id, sys_name, dim_key, float(dim_vals[i]))
            if dim_key not in metric_names:
                metric_names.append(dim_key)

    return metric_names


def compute_baselines_d2t(
    benchmark: D2TBenchmark, device: str = "cuda",
    max_samples: Optional[int] = None,
) -> List[str]:
    """Compute baseline scores for D2T, stored in benchmark data dict."""
    metric_names = []
    cands = benchmark.get_sys_summs()
    ref_summs_all = benchmark.get_ref_summs()
    doc_ids = benchmark.doc_ids

    if max_samples:
        cands = cands[:max_samples]
        ref_summs_all = ref_summs_all[:max_samples]
        doc_ids = doc_ids[:max_samples]

    # Use first reference for metrics that need a single ref
    single_refs = [refs[0] if refs else "" for refs in ref_summs_all]

    # BERTScore (max over references, following BartScore D2T)
    logger.info("  BERTScore...")
    bert = BERTScoreWrapper(device=device)
    n_refs = len(ref_summs_all[0]) if ref_summs_all else 1
    bert_p = np.zeros(len(cands))
    bert_r = np.zeros(len(cands))
    bert_f = np.zeros(len(cands))
    for ref_idx in range(n_refs):
        ref_list = [refs[ref_idx] if ref_idx < len(refs) else refs[0] for refs in ref_summs_all]
        scores = bert.score(cands, ref_list)
        bert_p = np.maximum(bert_p, scores["bertscore_p"])
        bert_r = np.maximum(bert_r, scores["bertscore_r"])
        bert_f = np.maximum(bert_f, scores["bertscore_f1"])
    for i, doc_id in enumerate(doc_ids):
        benchmark.store_metric_score(doc_id, "bertscore_f1", float(bert_f[i]))
        benchmark.store_metric_score(doc_id, "bertscore_p", float(bert_p[i]))
        benchmark.store_metric_score(doc_id, "bertscore_r", float(bert_r[i]))
    metric_names.extend(["bertscore_f1", "bertscore_p", "bertscore_r"])
    release_model(bert)
    del bert

    # BARTScore (max over references, following BartScore D2T)
    logger.info("  BARTScore...")
    bart = BARTScorerOfficial(device=device)
    for i, doc_id in enumerate(doc_ids):
        refs = ref_summs_all[i]
        sys_summ = cands[i]
        ref_hypo_scores = np.array(bart.score(refs, [sys_summ] * len(refs)))
        hypo_ref_scores = np.array(bart.score([sys_summ] * len(refs), refs))
        benchmark.store_metric_score(doc_id, "bartscore_ref_hypo", float(ref_hypo_scores.max()))
        benchmark.store_metric_score(doc_id, "bartscore_hypo_ref", float(hypo_ref_scores.max()))
        benchmark.store_metric_score(
            doc_id, "bartscore_avg_f",
            float((0.5 * (ref_hypo_scores + hypo_ref_scores)).max()),
        )
    metric_names.extend(["bartscore_ref_hypo", "bartscore_hypo_ref", "bartscore_avg_f"])
    release_model(bart)
    del bart

    # ROUGE (against first reference)
    logger.info("  ROUGE...")
    rouge = ROUGEScorer()
    rouge_scores = rouge.score(cands, single_refs)
    for i, doc_id in enumerate(doc_ids):
        for rk, rv in rouge_scores.items():
            benchmark.store_metric_score(doc_id, rk, float(rv[i]))
    metric_names.extend(list(rouge_scores.keys()))

    # MoverScore (against first reference)
    logger.info("  MoverScore...")
    mover = MoverScorer(device=device)
    mover_scores = mover.score(cands, single_refs)
    for i, doc_id in enumerate(doc_ids):
        benchmark.store_metric_score(doc_id, "moverscore", float(mover_scores[i]))
    metric_names.append("moverscore")
    release_model(mover)
    del mover

    # AlignScore (source=structured data, candidate=text)
    logger.info("  AlignScore...")
    srcs = benchmark.get_src_lines()
    if max_samples:
        srcs = srcs[:max_samples]
    align = AlignScorer(device=device)
    align_scores = align.score(srcs, cands)
    for i, doc_id in enumerate(doc_ids):
        benchmark.store_metric_score(doc_id, "alignscore", float(align_scores[i]))
    metric_names.append("alignscore")
    release_model(align)
    del align

    # QuestEval (source=structured data, candidate=text)
    logger.info("  QuestEval...")
    questeval = QuestEvalScorer(task="summarization", device=device)
    questeval_scores = questeval.score(srcs, cands)
    for i, doc_id in enumerate(doc_ids):
        benchmark.store_metric_score(doc_id, "questeval", float(questeval_scores[i]))
    metric_names.append("questeval")
    release_model(questeval)
    del questeval

    # UniEval (data2text, per-dimension + overall)
    logger.info("  UniEval...")
    unieval = UniEvalScorer(task="data2text", device=device)
    unieval_scores = unieval.score(srcs, cands, references=single_refs)
    for dim_key, dim_vals in unieval_scores.items():
        for i, doc_id in enumerate(doc_ids):
            benchmark.store_metric_score(doc_id, dim_key, float(dim_vals[i]))
        metric_names.append(dim_key)
    release_model(unieval)
    del unieval

    # G-Eval (LLM-as-Judge)
    logger.info("  G-Eval...")
    geval = GEvalScorer()
    geval_scores = geval.score(srcs, cands)
    for dim_key, dim_vals in geval_scores.items():
        for i, doc_id in enumerate(doc_ids):
            benchmark.store_metric_score(doc_id, dim_key, float(dim_vals[i]))
        metric_names.append(dim_key)

    return metric_names


# ---------------------------------------------------------------------------
# Full evaluation pipeline
# ---------------------------------------------------------------------------

def run_wmt_evaluation(
    model, bench_name: str, args, base_dir: str,
) -> Dict[str, Any]:
    """Run complete WMT evaluation for one language pair."""
    cfg = BENCHMARK_REGISTRY[bench_name]
    benchmark = load_benchmark(bench_name, base_dir)
    prompt_template = cfg["prompt_template"]

    logger.info(f"  Computing DiffScore...")
    ds_results = compute_diffscore_wmt(
        model, benchmark, prompt_template,
        K=args.K, T=args.T, batch_size=args.batch_size,
        scoring_mode=args.scoring_mode, max_samples=args.max_samples,
    )

    logger.info(f"  Computing baselines...")
    bl_results = compute_baselines_wmt(
        benchmark, device=args.device, max_samples=args.max_samples,
    )

    all_scores = {**ds_results, **bl_results}

    # Evaluate each metric
    eval_results = {}
    for metric_name, score_dict in all_scores.items():
        res = evaluate_wmt(score_dict["better"], score_dict["worse"])
        eval_results[metric_name] = res

    return eval_results


def run_sum_evaluation(
    model, bench_name: str, args, base_dir: str,
) -> Dict[str, Dict[str, Any]]:
    """Run complete SUM evaluation for one dataset."""
    cfg = BENCHMARK_REGISTRY[bench_name]
    benchmark = load_benchmark(bench_name, base_dir)
    prompt_template = cfg["prompt_template"]

    logger.info(f"  Computing DiffScore...")
    ds_metric_names = compute_diffscore_sum(
        model, benchmark, prompt_template,
        K=args.K, T=args.T, batch_size=args.batch_size,
        scoring_mode=args.scoring_mode, max_samples=args.max_samples,
    )

    logger.info(f"  Computing baselines...")
    bl_metric_names = compute_baselines_sum(
        benchmark, device=args.device, max_samples=args.max_samples,
    )

    all_metric_names = ds_metric_names + bl_metric_names

    # Evaluate per human metric
    eval_results = {}
    for human_metric in cfg["human_metrics"]:
        dim_results = {}
        for metric_name in all_metric_names:
            res = evaluate_sum(
                benchmark.data, metric_name, human_metric, cfg["eval_type"],
            )
            dim_results[metric_name] = res
        eval_results[human_metric] = dim_results

    return eval_results


def run_d2t_evaluation(
    model, bench_name: str, args, base_dir: str,
) -> Dict[str, Dict[str, Any]]:
    """Run complete D2T evaluation for one dataset."""
    cfg = BENCHMARK_REGISTRY[bench_name]
    benchmark = load_benchmark(bench_name, base_dir)
    prompt_template = cfg["prompt_template"]

    logger.info(f"  Computing DiffScore...")
    ds_metric_names = compute_diffscore_d2t(
        model, benchmark, prompt_template,
        K=args.K, T=args.T, batch_size=args.batch_size,
        scoring_mode=args.scoring_mode, max_samples=args.max_samples,
    )

    logger.info(f"  Computing baselines...")
    bl_metric_names = compute_baselines_d2t(
        benchmark, device=args.device, max_samples=args.max_samples,
    )

    all_metric_names = ds_metric_names + bl_metric_names

    # Evaluate per human metric
    eval_results = {}
    for human_metric in cfg["human_metrics"]:
        dim_results = {}
        for metric_name in all_metric_names:
            res = evaluate_d2t(benchmark.data, metric_name, human_metric)
            dim_results[metric_name] = res
        eval_results[human_metric] = dim_results

    return eval_results


# ---------------------------------------------------------------------------
# Main experiment runner
# ---------------------------------------------------------------------------

def run_experiment(args):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    os.makedirs(args.output_dir, exist_ok=True)
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # Parse benchmark selection
    if args.benchmarks:
        bench_names = args.benchmarks.split(",")
    elif args.tasks:
        bench_names = []
        for task in args.tasks.split(","):
            bench_names.extend(get_benchmarks_by_task(task.upper()))
    else:
        bench_names = get_all_benchmark_names()

    logger.info(f"Benchmarks to evaluate: {bench_names}")

    # Load model
    adapter_path = getattr(args, "adapter_path", None)
    model = MDLMWrapper(
        model_name=args.model_name, device=args.device, dtype=args.dtype,
        adapter_path=adapter_path,
    )

    all_results = {}

    for bench_name in bench_names:
        if bench_name not in BENCHMARK_REGISTRY:
            logger.warning(f"Unknown benchmark: {bench_name}, skipping")
            continue

        cfg = BENCHMARK_REGISTRY[bench_name]
        task = cfg["task"]

        logger.info(f"\n{'='*60}")
        logger.info(f"Benchmark: {bench_name} (task={task})")
        logger.info(f"{'='*60}")

        start_time = time.time()

        if task == "WMT":
            results = run_wmt_evaluation(model, bench_name, args, base_dir)
        elif task == "SUM":
            results = run_sum_evaluation(model, bench_name, args, base_dir)
        elif task == "D2T":
            results = run_d2t_evaluation(model, bench_name, args, base_dir)
        else:
            logger.warning(f"Unknown task: {task}")
            continue

        elapsed = time.time() - start_time
        logger.info(f"  Completed in {elapsed:.1f}s")

        all_results[bench_name] = results
        _print_summary(bench_name, results, task)

    # Release model
    release_model(model)
    del model

    # Save results
    out_path = os.path.join(args.output_dir, "meta_evaluation_results.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=float)
    logger.info(f"\nAll results saved to {out_path}")

    return all_results


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------

def _print_summary(bench_name: str, results: Dict, task: str):
    """Print formatted evaluation summary."""
    if task == "WMT":
        _print_wmt_summary(bench_name, results)
    elif task in ("SUM", "D2T"):
        _print_correlation_summary(bench_name, results, task)


def _print_wmt_summary(bench_name: str, results: Dict):
    """Print WMT Kendall τ table."""
    headers = ["Metric", "Kendall τ", "Accuracy"]
    rows = []
    scored = []
    for metric_name, res in results.items():
        tau = res.get("kendall_tau", 0)
        acc = res.get("accuracy", 0)
        scored.append((metric_name, tau, acc))
    scored.sort(key=lambda x: x[1], reverse=True)

    for m, tau, acc in scored[:25]:
        rows.append([m, f"{tau:.3f}", f"{acc:.3f}"])

    if rows:
        print(f"\n{bench_name} — WMT Kendall τ (top 25):")
        print(tabulate(rows, headers=headers, tablefmt="grid"))


def _print_correlation_summary(bench_name: str, results: Dict, task: str):
    """Print Spearman/Kendall table for SUM/D2T."""
    for dim_name, dim_results in results.items():
        headers = ["Metric", "Spearman ρ", "Kendall τ"]
        rows = []
        scored = []
        for metric_name, res in dim_results.items():
            sp = res.get("spearman", res.get("pearson", res.get("accuracy", 0)))
            kt = res.get("kendall_tau", sp)
            scored.append((metric_name, sp, kt))
        scored.sort(key=lambda x: x[1], reverse=True)

        for m, sp, kt in scored[:25]:
            rows.append([m, f"{sp:.3f}", f"{kt:.3f}"])

        if rows:
            print(f"\n{bench_name} — {dim_name} (top 25):")
            print(tabulate(rows, headers=headers, tablefmt="grid"))


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Experiment 1: Meta-Evaluation (BartScore-aligned)"
    )
    parser.add_argument("--model_name", type=str, default="GSAI-ML/LLaDA-8B-Instruct")
    parser.add_argument("--output_dir", type=str, default="outputs/exp1_meta_eval")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--adapter_path", type=str, default=None,
                        help="Path to fine-tuned LoRA adapter")
    parser.add_argument("--K", type=int, default=50)
    parser.add_argument("--T", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Limit samples per benchmark (for debugging)")
    parser.add_argument("--benchmarks", type=str, default=None,
                        help="Comma-separated benchmark names (e.g. wmt_de-en,sum_SummEval,d2t_BAGEL)")
    parser.add_argument("--tasks", type=str, default=None,
                        help="Comma-separated task categories (e.g. WMT,SUM,D2T)")
    parser.add_argument("--scoring_mode", type=str, default="mean_lp",
                        choices=["elbo", "mean_lp", "weighted"])
    args = parser.parse_args()

    run_experiment(args)
