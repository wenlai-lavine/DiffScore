"""
Experiment 6: Efficiency Analysis.

Reports timing benchmarks for DiffScore configurations:
- DiffScore-Fast (K=5, T=5)
- DiffScore-Standard (K=20, T=10)
- DiffScore-Full (K=50, T=10)

Compared against BARTScore wall-clock time.
Produces performance-efficiency Pareto frontier plot.
"""

import os
import sys
import json
import logging
import argparse
import time
import numpy as np
from scipy.stats import spearmanr
from tabulate import tabulate

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model import MDLMWrapper
from src.scoring import DiffScorer
from src.data.summeval import SummEvalDataset
from src.baselines.bartscore import BARTScorer
from src.visualization.quality_profile_plot import plot_pareto_frontier
from src.utils import release_model

logger = logging.getLogger(__name__)

CONFIGURATIONS = {
    "DiffScore-Fast": {"K": 5, "T": 5},
    "DiffScore-Standard": {"K": 20, "T": 10},
    "DiffScore-Full": {"K": 50, "T": 10},
}


def run_experiment(args):
    logging.basicConfig(level=logging.INFO)
    os.makedirs(args.output_dir, exist_ok=True)

    dataset = SummEvalDataset()
    if args.data_path:
        dataset.load_from_file(args.data_path)
    else:
        dataset.load_from_huggingface()

    sources = dataset.get_sources()
    candidates = dataset.get_candidates()
    human_scores = dataset.get_all_human_scores()
    prompt_template = dataset.get_prompt_template("default")

    n = args.max_samples or len(sources)
    sources = sources[:n]
    candidates = candidates[:n]
    human_scores = {k: v[:n] for k, v in human_scores.items()}

    model = MDLMWrapper(
        model_name=args.model_name, device=args.device, dtype=args.dtype,
    )

    results = {}

    # --- DiffScore configurations ---
    for config_name, config in CONFIGURATIONS.items():
        logger.info(f"Benchmarking {config_name} (K={config['K']}, T={config['T']})...")
        scorer = DiffScorer(
            model, K=config["K"], T=config["T"], batch_size=args.batch_size,
        )

        # Warmup
        scorer.score_conditional(sources[0], candidates[0], prompt_template)

        start = time.time()
        batch_results = scorer.score_batch(
            sources=sources, candidates=candidates,
            prompt_template=prompt_template, configuration="conditional",
        )
        elapsed = time.time() - start

        scores = np.array([r.scalar for r in batch_results])
        avg_rho = np.mean([
            spearmanr(scores, human_scores[dim])[0]
            for dim in human_scores
        ])

        results[config_name] = {
            "K": config["K"],
            "T": config["T"],
            "total_time_s": elapsed,
            "per_sample_time_s": elapsed / n,
            "n_samples": n,
            "n_forward_passes_per_sample": config["K"],
            "avg_spearman_rho": float(avg_rho),
        }
        logger.info(
            f"  {config_name}: {elapsed:.1f}s total, "
            f"{elapsed/n:.3f}s/sample, rho={avg_rho:.4f}"
        )

    # Release DiffScore model before loading BARTScore
    release_model(model)
    del model

    # --- BARTScore baseline ---
    logger.info("Benchmarking BARTScore...")
    bart = BARTScorer(device=args.device)

    start = time.time()
    bart_scores = bart.score(sources, candidates, configuration="conditional")
    bart_elapsed = time.time() - start

    bart_avg_rho = np.mean([
        spearmanr(bart_scores, human_scores[dim])[0]
        for dim in human_scores
    ])

    results["BARTScore"] = {
        "total_time_s": bart_elapsed,
        "per_sample_time_s": bart_elapsed / n,
        "n_samples": n,
        "n_forward_passes_per_sample": 1,
        "avg_spearman_rho": float(bart_avg_rho),
    }
    release_model(bart)
    del bart
    logger.info(
        f"  BARTScore: {bart_elapsed:.1f}s total, "
        f"{bart_elapsed/n:.3f}s/sample, rho={bart_avg_rho:.4f}"
    )

    # Print summary
    headers = ["Config", "K", "T", "Time/sample (s)", "Total (s)", "Avg ρ", "vs BART"]
    rows = []
    bart_time = results["BARTScore"]["per_sample_time_s"]
    for name, r in results.items():
        ratio = r["per_sample_time_s"] / bart_time if bart_time > 0 else float("inf")
        rows.append([
            name,
            r.get("K", "-"),
            r.get("T", "-"),
            f"{r['per_sample_time_s']:.4f}",
            f"{r['total_time_s']:.1f}",
            f"{r['avg_spearman_rho']:.4f}",
            f"{ratio:.1f}x",
        ])
    print("\nEfficiency Analysis:")
    print(tabulate(rows, headers=headers, tablefmt="grid"))

    # Pareto plot
    methods = list(results.keys())
    performance = [results[m]["avg_spearman_rho"] for m in methods]
    efficiency = [results[m]["per_sample_time_s"] for m in methods]

    plot_pareto_frontier(
        methods, performance, efficiency,
        save_path=os.path.join(args.output_dir, "pareto_frontier.pdf"),
    )

    out_path = os.path.join(args.output_dir, "efficiency_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=float)
    logger.info(f"Results saved to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Experiment 6: Efficiency Analysis")
    parser.add_argument("--model_name", type=str, default="GSAI-ML/LLaDA-8B-Instruct")
    parser.add_argument("--data_path", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="outputs/exp6_efficiency")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_samples", type=int, default=100)
    args = parser.parse_args()

    run_experiment(args)
