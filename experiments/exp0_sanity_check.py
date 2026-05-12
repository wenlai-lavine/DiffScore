"""
Experiment 0: Sanity Check (Pre-validation, not for the paper).

Validates:
1. Conditional scoring correlates significantly with human scores (> random)
2. Random source text control: replacing source with random document degrades score
3. Prompt template comparison: test 3-5 templates, select best
4. MC sampling convergence: plot Spearman rho vs. K
"""

import os
import sys
import json
import logging
import argparse
import numpy as np
from scipy.stats import spearmanr
from tabulate import tabulate

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model import MDLMWrapper
from src.scoring import DiffScorer
from src.data.summeval import SummEvalDataset, PROMPT_TEMPLATES
from src.evaluation.correlation import compute_correlations
from src.visualization.quality_profile_plot import plot_convergence_curve
from src.utils import release_model

logger = logging.getLogger(__name__)


def run_sanity_check(args):
    logging.basicConfig(level=logging.INFO)
    os.makedirs(args.output_dir, exist_ok=True)

    # Load model
    model = MDLMWrapper(
        model_name=args.model_name,
        device=args.device,
        dtype=args.dtype,
    )

    # Load data
    dataset = SummEvalDataset()
    if args.data_path:
        dataset.load_from_file(args.data_path)
    else:
        dataset.load_from_huggingface()

    sources = dataset.get_sources()
    candidates = dataset.get_candidates()
    human_scores = dataset.get_all_human_scores()

    if args.max_samples:
        sources = sources[:args.max_samples]
        candidates = candidates[:args.max_samples]
        human_scores = {k: v[:args.max_samples] for k, v in human_scores.items()}

    results = {}

    # --- Test 1: Conditional scoring validity ---
    logger.info("=== Test 1: Conditional Scoring Validity ===")
    template = dataset.get_prompt_template("instruct")
    scorer = DiffScorer(
        model, K=args.K, T=args.T, batch_size=args.batch_size,
        scoring_mode=getattr(args, "scoring_mode", "elbo"),
    )

    cond_results = scorer.score_batch(
        sources=sources, candidates=candidates,
        prompt_template=template, configuration="conditional",
    )
    cond_scores = np.array([r.scalar for r in cond_results])

    test1_results = {}
    for dim in human_scores:
        corr = compute_correlations(cond_scores, human_scores[dim], level="segment")
        test1_results[dim] = corr
        rho = corr.get("spearman_rho", (0, 1))
        logger.info(f"  {dim}: Spearman rho={rho[0]:.4f}, p={rho[1]:.6f}")

    results["test1_conditional_validity"] = {
        dim: {k: (float(v[0]), float(v[1])) for k, v in corrs.items()}
        for dim, corrs in test1_results.items()
    }

    # --- Test 2: Random source control ---
    logger.info("=== Test 2: Random Source Control ===")
    import random
    rng = random.Random(42)
    random_sources = sources.copy()
    rng.shuffle(random_sources)
    for i in range(len(random_sources)):
        if random_sources[i] == sources[i]:
            j = (i + 1) % len(random_sources)
            random_sources[i], random_sources[j] = random_sources[j], random_sources[i]

    rand_results = scorer.score_batch(
        sources=random_sources, candidates=candidates,
        prompt_template=template, configuration="conditional",
    )
    rand_scores = np.array([r.scalar for r in rand_results])

    score_diff = cond_scores.mean() - rand_scores.mean()
    logger.info(f"  Mean score (correct source): {cond_scores.mean():.4f}")
    logger.info(f"  Mean score (random source):  {rand_scores.mean():.4f}")
    logger.info(f"  Difference: {score_diff:.4f}")

    results["test2_random_control"] = {
        "correct_source_mean": float(cond_scores.mean()),
        "random_source_mean": float(rand_scores.mean()),
        "difference": float(score_diff),
    }

    # --- Test 3: Prompt template comparison ---
    logger.info("=== Test 3: Prompt Template Comparison ===")
    template_results = {}
    for tname, tmpl in PROMPT_TEMPLATES.items():
        t_results = scorer.score_batch(
            sources=sources, candidates=candidates,
            prompt_template=tmpl, configuration="conditional",
        )
        t_scores = np.array([r.scalar for r in t_results])

        t_corrs = {}
        for dim in human_scores:
            rho, p = spearmanr(t_scores, human_scores[dim])
            t_corrs[dim] = float(rho)
        template_results[tname] = t_corrs
        avg_rho = np.mean(list(t_corrs.values()))
        logger.info(f"  Template '{tname}': avg Spearman rho = {avg_rho:.4f}")

    results["test3_prompt_templates"] = template_results

    # --- Test 4: MC convergence ---
    logger.info("=== Test 4: MC Sampling Convergence ===")
    K_values = [5, 10, 20, 50]
    convergence = {dim: [] for dim in human_scores}

    for K in K_values:
        k_scorer = DiffScorer(model, K=K, T=args.T, batch_size=args.batch_size)
        k_results = k_scorer.score_batch(
            sources=sources, candidates=candidates,
            prompt_template=template, configuration="conditional",
        )
        k_scores = np.array([r.scalar for r in k_results])

        for dim in human_scores:
            rho, _ = spearmanr(k_scores, human_scores[dim])
            convergence[dim].append(float(rho))
        logger.info(f"  K={K}: avg rho = {np.mean([convergence[d][-1] for d in convergence]):.4f}")

    results["test4_convergence"] = {"K_values": K_values, "correlations": convergence}

    plot_convergence_curve(
        K_values, convergence,
        save_path=os.path.join(args.output_dir, "mc_convergence.pdf"),
    )

    # Save results
    out_path = os.path.join(args.output_dir, "sanity_check_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved to {out_path}")

    # Decision gate
    best_template = max(
        template_results.items(),
        key=lambda x: np.mean(list(x[1].values()))
    )
    logger.info(f"\n=== DECISION ===")
    logger.info(f"Best prompt template: '{best_template[0]}'")

    any_significant = any(
        test1_results[dim].get("spearman_rho", (0, 1))[1] < 0.05
        for dim in test1_results
    )
    if any_significant and score_diff > 0:
        logger.info("PASS: Conditional scoring is valid. Proceed to main experiments.")
    else:
        logger.warning(
            "FAIL: Conditional scoring may not be effective. "
            "Consider fine-tuning the model on paired data."
        )

    # Release model GPU memory
    release_model(model)
    del model, scorer


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Experiment 0: Sanity Check")
    parser.add_argument("--model_name", type=str, default="GSAI-ML/LLaDA-8B-Instruct")
    parser.add_argument("--data_path", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="outputs/exp0_sanity_check")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--K", type=int, default=20)
    parser.add_argument("--T", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_samples", type=int, default=None)
    args = parser.parse_args()

    run_sanity_check(args)
