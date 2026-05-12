"""
Experiment 5: Ablation Study.

Ablation dimensions:
1. MC samples K: {5, 10, 20, 50}
2. Timesteps T: {1, 5, 10, 20}
3. Model: LLaDA-8B-Instruct vs Base vs Dream-7B
4. Normalization: none vs length vs IDF-weighted
5. Prompt template: 3-5 variants
6. Masking strategy: random vs entity-mask vs content-mask
"""

import os
import sys
import json
import logging
import argparse
import numpy as np
from scipy.stats import spearmanr
from tabulate import tabulate
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model import MDLMWrapper
from src.scoring import DiffScorer
from src.model.masking import RandomMasker
from src.data.summeval import SummEvalDataset, PROMPT_TEMPLATES
from src.evaluation.correlation import compute_correlations
from src.utils import release_model

logger = logging.getLogger(__name__)


def _compute_scores(scorer, sources, candidates, prompt_template):
    """Helper to compute conditional scores and return scalar array."""
    results = scorer.score_batch(
        sources=sources, candidates=candidates,
        prompt_template=prompt_template, configuration="conditional",
    )
    return np.array([r.scalar for r in results])


def ablation_K(model, dataset, prompt_template, args, human_scores):
    """Ablation over MC sample count K."""
    logger.info("=== Ablation: K (MC samples) ===")
    K_values = [5, 10, 20, 50]
    sources = dataset.get_sources()[:args.max_samples]
    candidates = dataset.get_candidates()[:args.max_samples]

    results = {}
    for K in K_values:
        scorer = DiffScorer(model, K=K, T=args.T, batch_size=args.batch_size)
        scores = _compute_scores(scorer, sources, candidates, prompt_template)
        corrs = {}
        for dim in human_scores:
            rho, _ = spearmanr(scores, human_scores[dim][:args.max_samples])
            corrs[dim] = float(rho)
        results[K] = corrs
        logger.info(f"  K={K}: avg rho = {np.mean(list(corrs.values())):.4f}")

    return results


def ablation_T(model, dataset, prompt_template, args, human_scores):
    """Ablation over timestep count T."""
    logger.info("=== Ablation: T (timesteps) ===")
    T_values = [1, 5, 10, 20]
    sources = dataset.get_sources()[:args.max_samples]
    candidates = dataset.get_candidates()[:args.max_samples]

    results = {}
    for T in T_values:
        scorer = DiffScorer(model, K=args.K, T=T, batch_size=args.batch_size)
        scores = _compute_scores(scorer, sources, candidates, prompt_template)
        corrs = {}
        for dim in human_scores:
            rho, _ = spearmanr(scores, human_scores[dim][:args.max_samples])
            corrs[dim] = float(rho)
        results[T] = corrs
        logger.info(f"  T={T}: avg rho = {np.mean(list(corrs.values())):.4f}")

    return results


def ablation_prompt(model, dataset, args, human_scores):
    """Ablation over prompt templates."""
    logger.info("=== Ablation: Prompt Templates ===")
    sources = dataset.get_sources()[:args.max_samples]
    candidates = dataset.get_candidates()[:args.max_samples]

    scorer = DiffScorer(model, K=args.K, T=args.T, batch_size=args.batch_size)
    results = {}
    for tname, tmpl in PROMPT_TEMPLATES.items():
        scores = _compute_scores(scorer, sources, candidates, tmpl)
        corrs = {}
        for dim in human_scores:
            rho, _ = spearmanr(scores, human_scores[dim][:args.max_samples])
            corrs[dim] = float(rho)
        results[tname] = corrs
        logger.info(f"  Template '{tname}': avg rho = {np.mean(list(corrs.values())):.4f}")

    return results


def ablation_masking(model, dataset, prompt_template, args, human_scores):
    """Ablation over masking strategies."""
    logger.info("=== Ablation: Masking Strategy ===")
    sources = dataset.get_sources()[:args.max_samples]
    candidates = dataset.get_candidates()[:args.max_samples]

    strategies = {"random": RandomMasker()}

    try:
        from src.model.masking import EntityMasker, ContentMasker
        strategies["entity"] = EntityMasker(model.tokenizer)
        strategies["content"] = ContentMasker(model.tokenizer)
    except Exception as e:
        logger.warning(f"Structured masking unavailable: {e}")

    results = {}
    for sname, masker in strategies.items():
        scorer = DiffScorer(
            model, K=args.K, T=args.T, batch_size=args.batch_size, masker=masker,
        )
        scores = _compute_scores(scorer, sources, candidates, prompt_template)
        corrs = {}
        for dim in human_scores:
            rho, _ = spearmanr(scores, human_scores[dim][:args.max_samples])
            corrs[dim] = float(rho)
        results[sname] = corrs
        logger.info(f"  Strategy '{sname}': avg rho = {np.mean(list(corrs.values())):.4f}")

    return results


def run_experiment(args):
    logging.basicConfig(level=logging.INFO)
    os.makedirs(args.output_dir, exist_ok=True)

    model = MDLMWrapper(
        model_name=args.model_name, device=args.device, dtype=args.dtype,
    )

    dataset = SummEvalDataset()
    if args.data_path:
        dataset.load_from_file(args.data_path)
    else:
        dataset.load_from_huggingface()

    human_scores = dataset.get_all_human_scores()
    prompt_template = dataset.get_prompt_template("default")

    all_results = {}

    if "K" in args.ablations:
        all_results["K"] = ablation_K(model, dataset, prompt_template, args, human_scores)

    if "T" in args.ablations:
        all_results["T"] = ablation_T(model, dataset, prompt_template, args, human_scores)

    if "prompt" in args.ablations:
        all_results["prompt"] = ablation_prompt(model, dataset, args, human_scores)

    if "masking" in args.ablations:
        all_results["masking"] = ablation_masking(
            model, dataset, prompt_template, args, human_scores,
        )

    # Print summary tables
    for abl_name, abl_results in all_results.items():
        headers = ["Setting"] + list(human_scores.keys()) + ["Average"]
        rows = []
        for setting, corrs in abl_results.items():
            vals = list(corrs.values())
            rows.append([setting] + [f"{v:.4f}" for v in vals] + [f"{np.mean(vals):.4f}"])
        print(f"\nAblation: {abl_name}")
        print(tabulate(rows, headers=headers, tablefmt="grid"))

    out_path = os.path.join(args.output_dir, "ablation_results.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    logger.info(f"Results saved to {out_path}")

    release_model(model)
    del model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Experiment 5: Ablation")
    parser.add_argument("--model_name", type=str, default="GSAI-ML/LLaDA-8B-Instruct")
    parser.add_argument("--data_path", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="outputs/exp5_ablation")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--K", type=int, default=50)
    parser.add_argument("--T", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_samples", type=int, default=200)
    parser.add_argument("--ablations", type=str, nargs="+",
                        default=["K", "T", "prompt", "masking"],
                        help="Which ablations to run")
    args = parser.parse_args()

    run_experiment(args)
