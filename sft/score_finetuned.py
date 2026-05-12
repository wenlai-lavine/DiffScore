"""
Score benchmarks with a fine-tuned DiffScore model (DiffScore-FT).

The fine-tuned model is loaded via MDLMWrapper's adapter_path parameter,
which loads a LoRA checkpoint and merges it into the base model. The
scoring logic is identical to zero-shot DiffScore — the only difference
is the underlying model weights.

This parallels how BARTScore uses BART-large-CNN: same scoring mechanism
(generation probability), different model weights (fine-tuned on CNN/DM).
For DiffScore-FT: same scoring mechanism (masked reconstruction probability),
different model weights (fine-tuned on CNN/DM via masked reconstruction).

Usage:
------
# Score with DiffScore-FT:
python sft/score_finetuned.py \
--adapter_path /home/ubuntu-1/shenyingli/DiffScore_SFT_Model/llada_wmt_sft/checkpoint-final \
--benchmarks sum_Newsroom sum_QAGS_CNN \
--output_dir outputs/diffscore_ft


benchmarks: wmt_de-en wmt_fi-en wmt_gu-en wmt_kk-en wmt_lt-en wmt_ru-en wmt_zh-en sum_Newsroom sum_QAGS_CNN sum_QAGS_XSUM sum_Rank19 sum_REALSumm sum_SummEval d2t_BAGEL d2t_SFHOT d2t_SFRES

# Score with base model (DiffScore-Zero, for comparison):
    python sft/score_finetuned.py \
        --base_model GSAI-ML/LLaDA-8B-Base \
        --benchmarks sum_SummEval \
        --output_dir outputs/diffscore_zero

"""

import os
import sys
import json
import argparse
from pathlib import Path

import torch
import numpy as np
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.model.mdlm_wrapper import MDLMWrapper
from src.scoring.diffscore import DiffScorer
from src.scoring.pmi import PMIScorer
from src.data.bartscore_benchmark import (
    BENCHMARK_REGISTRY,
    load_benchmark,
    get_benchmarks_by_task,
)


def score_sum_benchmark(
    scorer: DiffScorer,
    benchmark_name: str,
    base_dir: str = ".",
    configs: list[str] = ("conditional", "marginal"),
) -> dict:
    """Score a SUM benchmark and compute correlations with human judgments."""
    bm = load_benchmark(benchmark_name, base_dir=base_dir)
    cfg = BENCHMARK_REGISTRY[benchmark_name]
    prompt = cfg["prompt_template"]
    results = {}

    for config_name in configs:
        all_scores = []
        all_human = {m: [] for m in bm.human_metrics}

        for doc_id, sys_name, src, sys_summ in bm.iter_all_samples():
            if config_name == "conditional":
                result = scorer.score_conditional(src, sys_summ, prompt)
            elif config_name == "marginal":
                result = scorer.score_marginal(sys_summ)
            elif config_name == "reverse":
                result = scorer.score_reverse(src, sys_summ, prompt)
            elif config_name == "bidirectional":
                result = scorer.score_bidirectional(src, sys_summ, prompt)
            else:
                raise ValueError(f"Unknown config: {config_name}")

            all_scores.append(result.scalar)
            for m in bm.human_metrics:
                human_score = bm.data[doc_id]["sys_summs"][sys_name]["scores"].get(m, 0)
                all_human[m].append(human_score)

        results[config_name] = {}
        for m in bm.human_metrics:
            rho, p = stats.spearmanr(all_scores, all_human[m])
            tau, p_tau = stats.kendalltau(all_scores, all_human[m])
            results[config_name][m] = {
                "spearman_rho": round(rho, 4),
                "spearman_p": round(p, 6),
                "kendall_tau": round(tau, 4),
                "kendall_p": round(p_tau, 6),
                "n_samples": len(all_scores),
            }
            print(f"  [{benchmark_name}] {config_name}/{m}: rho={rho:.4f}, tau={tau:.4f}")

    return results


def score_wmt_benchmark(
    scorer: DiffScorer,
    benchmark_name: str,
    base_dir: str = ".",
) -> dict:
    """Score a WMT benchmark with preference accuracy."""
    bm = load_benchmark(benchmark_name, base_dir=base_dir)
    cfg = BENCHMARK_REGISTRY[benchmark_name]
    prompt = cfg["prompt_template"]

    concordant = 0
    total = 0

    for doc_id in bm.doc_ids:
        entry = bm.data[doc_id]
        src = entry.get("ref", entry.get("src", ""))
        better = entry["better"]["sys"]
        worse = entry["worse"]["sys"]

        score_better = scorer.score_conditional(src, better, prompt).scalar
        score_worse = scorer.score_conditional(src, worse, prompt).scalar

        if score_better > score_worse:
            concordant += 1
        total += 1

    tau = concordant / max(1, total)
    print(f"  [{benchmark_name}] concordance tau = {tau:.4f} ({concordant}/{total})")
    return {"concordance_tau": round(tau, 4), "concordant": concordant, "total": total}


def score_d2t_benchmark(
    scorer: DiffScorer,
    benchmark_name: str,
    base_dir: str = ".",
) -> dict:
    """Score a D2T benchmark."""
    bm = load_benchmark(benchmark_name, base_dir=base_dir)
    cfg = BENCHMARK_REGISTRY[benchmark_name]
    prompt = cfg["prompt_template"]

    all_scores = []
    srcs = bm.get_src_lines()
    sys_summs = bm.get_sys_summs()

    for src, sys_summ in zip(srcs, sys_summs):
        result = scorer.score_conditional(src, sys_summ, prompt)
        all_scores.append(result.scalar)

    results = {}
    for m in bm.human_metrics:
        human_scores = bm.get_human_scores(m)
        rho, p = stats.spearmanr(all_scores, human_scores)
        tau, p_tau = stats.kendalltau(all_scores, human_scores)
        results[m] = {
            "spearman_rho": round(rho, 4),
            "kendall_tau": round(tau, 4),
            "n_samples": len(all_scores),
        }
        print(f"  [{benchmark_name}] {m}: rho={rho:.4f}, tau={tau:.4f}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Score benchmarks with DiffScore-FT")
    parser.add_argument("--base_model", default="GSAI-ML/LLaDA-8B-Base")
    parser.add_argument("--adapter_path", default=None, help="Path to LoRA adapter checkpoint")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument(
        "--benchmarks", nargs="+",
        default=["sum_SummEval"],
    )
    parser.add_argument("--output_dir", default="outputs/diffscore_ft")
    parser.add_argument("--K", type=int, default=20)
    parser.add_argument("--T", type=int, default=10)
    parser.add_argument("--scoring_mode", default="mean_lp")
    parser.add_argument("--base_dir", default=str(PROJECT_ROOT))
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    model_label = "DiffScore-FT" if args.adapter_path else "DiffScore-Zero"
    print(f"Loading model: {args.base_model} ({model_label})")
    if args.adapter_path:
        print(f"  Adapter: {args.adapter_path}")

    wrapper = MDLMWrapper(
        model_name=args.base_model,
        device=args.device,
        dtype=args.dtype,
        adapter_path=args.adapter_path,
    )

    scorer = DiffScorer(
        model=wrapper,
        K=args.K,
        T=args.T,
        scoring_mode=args.scoring_mode,
    )

    all_results = {"model": model_label, "adapter": args.adapter_path, "benchmarks": {}}
    for bm_name in args.benchmarks:
        print(f"\nScoring {bm_name}...")
        cfg = BENCHMARK_REGISTRY[bm_name]
        task = cfg["task"]

        if task == "SUM":
            results = score_sum_benchmark(scorer, bm_name, args.base_dir)
        elif task == "WMT":
            results = score_wmt_benchmark(scorer, bm_name, args.base_dir)
        elif task == "D2T":
            results = score_d2t_benchmark(scorer, bm_name, args.base_dir)
        else:
            print(f"  Unknown task type: {task}, skipping")
            continue

        all_results["benchmarks"][bm_name] = results

    output_path = os.path.join(args.output_dir, f"{model_label.lower()}_results.json")
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
