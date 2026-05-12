"""
Experiment 7: Interpretability Case Study.

1. Token-level quality heatmaps for selected summaries
2. Quality profile curves comparing high vs low quality texts
3. Token-level DiffScore vs MQM error annotations (on WMT data)
"""

import os
import sys
import json
import logging
import argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model import MDLMWrapper
from src.scoring import DiffScorer, TokenLevelScorer
from src.data.summeval import SummEvalDataset
from src.visualization.heatmap import plot_token_heatmap
from src.visualization.quality_profile_plot import plot_quality_profiles
from src.utils import release_model

logger = logging.getLogger(__name__)


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

    sources = dataset.get_sources()
    candidates = dataset.get_candidates()
    human_scores = dataset.get_all_human_scores()
    prompt_template = dataset.get_prompt_template("default")

    # --- Case 1: Token-level heatmaps ---
    logger.info("=== Token-Level Quality Heatmaps ===")
    token_scorer = TokenLevelScorer(model, K=args.K, T=args.T)

    # Select representative examples
    coherence = human_scores["coherence"]
    n_samples = min(len(candidates), args.max_samples or len(candidates))

    # High quality example
    high_idx = np.argmax(coherence[:n_samples])
    # Low quality example
    low_idx = np.argmin(coherence[:n_samples])
    # Median quality
    median_val = np.median(coherence[:n_samples])
    med_idx = np.argmin(np.abs(coherence[:n_samples] - median_val))

    cases = {"high_quality": high_idx, "low_quality": low_idx, "median_quality": med_idx}

    for case_name, idx in cases.items():
        logger.info(f"  Processing {case_name} example (idx={idx})...")

        token_result = token_scorer.score_tokens(
            candidates[idx], context=sources[idx], prompt_template=prompt_template,
        )

        plot_token_heatmap(
            token_result.tokens,
            token_result.scores,
            title=f"Token Quality: {case_name} (coherence={coherence[idx]:.1f})",
            save_path=os.path.join(args.output_dir, f"token_heatmap_{case_name}.pdf"),
        )

        # Save case info
        case_info = {
            "index": int(idx),
            "source_preview": sources[idx][:200],
            "candidate": candidates[idx],
            "human_scores": {dim: float(human_scores[dim][idx]) for dim in human_scores},
            "tokens": token_result.tokens,
            "token_scores": token_result.scores.tolist(),
        }
        with open(os.path.join(args.output_dir, f"case_{case_name}.json"), "w") as f:
            json.dump(case_info, f, indent=2)

    # --- Case 2: Quality profile curves ---
    logger.info("=== Quality Profile Curves ===")
    scorer = DiffScorer(model, K=args.K, T=args.T, batch_size=args.batch_size)

    profiles_to_plot = {}
    for case_name, idx in cases.items():
        result = scorer.score_conditional(
            sources[idx], candidates[idx], prompt_template, return_profile=True,
        )
        profiles_to_plot[f"{case_name} (coh={coherence[idx]:.1f})"] = result.profile

    plot_quality_profiles(
        profiles_to_plot,
        title="Quality Profiles: Selected Examples",
        save_path=os.path.join(args.output_dir, "quality_profiles_cases.pdf"),
    )

    # --- Case 3: Compare marginal vs conditional profiles ---
    logger.info("=== Marginal vs Conditional Profile ===")
    for case_name, idx in [("high_quality", high_idx), ("low_quality", low_idx)]:
        mar_result = scorer.score_marginal(candidates[idx], return_profile=True)
        cond_result = scorer.score_conditional(
            sources[idx], candidates[idx], prompt_template, return_profile=True,
        )

        # Compute per-timestep PMI
        pmi_profile = {}
        for t in cond_result.profile:
            pmi_profile[t] = cond_result.profile[t] - mar_result.profile.get(t, 0)

        plot_quality_profiles(
            {
                "Conditional (overall)": cond_result.profile,
                "Marginal (fluency)": mar_result.profile,
                "PMI (relevance)": pmi_profile,
            },
            title=f"Score Decomposition: {case_name}",
            save_path=os.path.join(args.output_dir, f"decomposition_{case_name}.pdf"),
        )

    logger.info(f"All case study outputs saved to {args.output_dir}")

    release_model(model)
    del model, scorer, token_scorer


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Experiment 7: Case Study")
    parser.add_argument("--model_name", type=str, default="GSAI-ML/LLaDA-8B-Instruct")
    parser.add_argument("--data_path", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="outputs/exp7_case_study")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--K", type=int, default=50)
    parser.add_argument("--T", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_samples", type=int, default=200)
    args = parser.parse_args()

    run_experiment(args)
