"""
Experiment 2: Multi-Timestep Quality Profile Analysis.

Verifies the core hypothesis: different evaluation dimensions correlate
best with DiffScore at different timesteps.
- Low t (rich context) -> Fluency
- High t (sparse context) -> Coherence

Produces:
1. Dimension x Timestep correlation heatmap
2. Learned vs uniform weight comparison
"""

import os
import sys
import json
import logging
import argparse
import numpy as np
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model import MDLMWrapper
from src.scoring import DiffScorer, QualityProfiler
from src.data.summeval import SummEvalDataset
from src.visualization.heatmap import plot_timestep_dimension_heatmap
from src.visualization.quality_profile_plot import plot_quality_profiles
from src.utils import release_model

logger = logging.getLogger(__name__)


def run_experiment(args):
    logging.basicConfig(level=logging.INFO)
    os.makedirs(args.output_dir, exist_ok=True)

    model = MDLMWrapper(
        model_name=args.model_name, device=args.device, dtype=args.dtype,
    )
    scorer = DiffScorer(model, K=args.K, T=args.T, batch_size=args.batch_size)
    profiler = QualityProfiler(T=args.T)

    dataset = SummEvalDataset()
    if args.data_path:
        dataset.load_from_file(args.data_path)
    else:
        dataset.load_from_huggingface()

    sources = dataset.get_sources()
    candidates = dataset.get_candidates()
    human_scores = dataset.get_all_human_scores()
    prompt_template = dataset.get_prompt_template(args.prompt_template)

    if args.max_samples:
        sources = sources[:args.max_samples]
        candidates = candidates[:args.max_samples]
        human_scores = {k: v[:args.max_samples] for k, v in human_scores.items()}

    # Compute per-timestep scores (conditional configuration)
    logger.info("Computing per-timestep conditional scores...")
    cond_results = scorer.score_batch(
        sources=sources, candidates=candidates,
        prompt_template=prompt_template, configuration="conditional",
    )

    # Extract profiles
    profiles = profiler.extract_profiles_batch(cond_results)

    # --- Analysis 1: Dimension x Timestep correlation matrix ---
    logger.info("Computing dimension x timestep correlation matrix...")
    corr_matrix = profiler.dimension_timestep_matrix(
        profiles, human_scores, method="spearman",
    )

    plot_timestep_dimension_heatmap(
        corr_matrix,
        title="DiffScore: Dimension × Timestep Correlation (SummEval)",
        save_path=os.path.join(args.output_dir, "dim_timestep_heatmap.pdf"),
    )

    # Log key findings
    for dim, corrs in corr_matrix.items():
        best_t = max(corrs, key=corrs.get)
        logger.info(f"  {dim}: best timestep t={best_t:.1f} (rho={corrs[best_t]:.4f})")

    # --- Analysis 2: Learned weights vs uniform ---
    logger.info("Learning optimal timestep weights (with cross-validation)...")
    n = len(candidates)

    weight_results = {}
    for dim in human_scores:
        from sklearn.model_selection import KFold

        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        rho_learned_list = []
        rho_uniform_list = []
        all_learned_weights = []

        for train_idx, test_idx in kf.split(profiles):
            train_profiles = profiles[train_idx]
            test_profiles = profiles[test_idx]
            train_human = human_scores[dim][train_idx]
            test_human = human_scores[dim][test_idx]

            learned_weights = profiler.learn_weights(
                train_profiles, train_human, method="spearman",
                n_restarts=10, cv_folds=0,
            )
            all_learned_weights.append(learned_weights)

            learned_scores = np.array([
                profiler.aggregate_weighted(p, learned_weights)
                for p in test_profiles
            ])
            uniform_scores = np.array([
                profiler.aggregate_uniform(p) for p in test_profiles
            ])

            rho_l, _ = spearmanr(learned_scores, test_human)
            rho_u, _ = spearmanr(uniform_scores, test_human)
            rho_learned_list.append(rho_l)
            rho_uniform_list.append(rho_u)

        mean_weights = np.mean(all_learned_weights, axis=0)
        mean_weights = mean_weights / mean_weights.sum()
        rho_learned = float(np.mean(rho_learned_list))
        rho_uniform = float(np.mean(rho_uniform_list))

        weight_results[dim] = {
            "learned_weights": mean_weights.tolist(),
            "rho_learned": rho_learned,
            "rho_learned_std": float(np.std(rho_learned_list)),
            "rho_uniform": rho_uniform,
            "rho_uniform_std": float(np.std(rho_uniform_list)),
            "improvement": float(rho_learned - rho_uniform),
            "per_fold_rho_learned": [float(r) for r in rho_learned_list],
            "per_fold_rho_uniform": [float(r) for r in rho_uniform_list],
        }
        logger.info(
            f"  {dim}: learned rho={rho_learned:.4f} ± {np.std(rho_learned_list):.4f}, "
            f"uniform rho={rho_uniform:.4f}, "
            f"delta={rho_learned - rho_uniform:+.4f}"
        )

    # --- Analysis 3: Quality profile curves for high/low quality ---
    logger.info("Generating quality profile comparison...")
    overall = human_scores.get("coherence", human_scores.get(
        list(human_scores.keys())[0]))
    median = np.median(overall)
    high_mask = overall[: len(candidates)] >= np.percentile(overall[: len(candidates)], 75)
    low_mask = overall[: len(candidates)] <= np.percentile(overall[: len(candidates)], 25)

    timesteps = profiler.timesteps
    high_profile = {t: float(profiles[high_mask, k].mean())
                    for k, t in enumerate(timesteps)}
    low_profile = {t: float(profiles[low_mask, k].mean())
                   for k, t in enumerate(timesteps)}

    plot_quality_profiles(
        {"High Quality (top 25%)": high_profile, "Low Quality (bottom 25%)": low_profile},
        title="Quality Profiles: High vs Low Quality Summaries",
        save_path=os.path.join(args.output_dir, "quality_profile_comparison.pdf"),
    )

    # Save all results
    results = {
        "correlation_matrix": {
            dim: {str(t): float(v) for t, v in corrs.items()}
            for dim, corrs in corr_matrix.items()
        },
        "weight_learning": weight_results,
    }

    out_path = os.path.join(args.output_dir, "quality_profile_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved to {out_path}")

    release_model(model)
    del model, scorer


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Experiment 2: Quality Profile")
    parser.add_argument("--model_name", type=str, default="GSAI-ML/LLaDA-8B-Instruct")
    parser.add_argument("--data_path", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="outputs/exp2_quality_profile")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--K", type=int, default=50)
    parser.add_argument("--T", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--prompt_template", type=str, default="default")
    args = parser.parse_args()

    run_experiment(args)
