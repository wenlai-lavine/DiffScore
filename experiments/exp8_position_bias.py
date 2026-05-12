"""
Experiment 8: Position Bias Empirical Analysis.

Computes PosBias(n) = Std_j[log p(x_n^(j) | context_n^(j))]
for DiffScore and BARTScore, comparing uniformity across positions.

DiffScore should show more uniform position bias thanks to
random masking over all positions.
"""

import os
import sys
import json
import logging
import argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model import MDLMWrapper
from src.analysis.position_bias import PositionBiasAnalyzer
from src.data.summeval import SummEvalDataset
from src.visualization.position_bias_plot import (
    plot_position_bias, plot_position_bias_distribution,
)

logger = logging.getLogger(__name__)


def run_experiment(args):
    logging.basicConfig(level=logging.INFO)
    os.makedirs(args.output_dir, exist_ok=True)

    dataset = SummEvalDataset()
    if args.data_path:
        dataset.load_from_file(args.data_path)
    else:
        dataset.load_from_huggingface()

    candidates = dataset.get_candidates()
    if args.max_samples:
        candidates = candidates[:args.max_samples]

    analyzer = PositionBiasAnalyzer(max_positions=args.max_positions)
    bias_curves = {}

    # --- DiffScore position bias ---
    logger.info("Computing DiffScore position bias...")
    mdlm = MDLMWrapper(
        model_name=args.model_name, device=args.device, dtype=args.dtype,
    )

    diff_bias = analyzer.compute_diffscore_position_bias(
        mdlm, candidates, K=args.K, T=args.T,
    )
    bias_curves["DiffScore (LLaDA-8B)"] = diff_bias

    diff_uniformity = analyzer.compute_uniformity_metric(diff_bias)
    logger.info(f"  DiffScore uniformity (CoV): {diff_uniformity:.4f}")

    # Free memory
    del mdlm
    import torch
    torch.cuda.empty_cache()

    # --- AR model position bias ---
    logger.info("Computing AR model position bias...")
    from transformers import AutoModelForCausalLM, AutoTokenizer

    ar_name = args.ar_model_name
    ar_tokenizer = AutoTokenizer.from_pretrained(ar_name)
    ar_model = AutoModelForCausalLM.from_pretrained(
        ar_name, torch_dtype=torch.bfloat16,
    ).to(args.device)
    ar_model.eval()

    ar_bias = analyzer.compute_ar_position_bias(
        ar_model, ar_tokenizer, candidates,
    )
    bias_curves[f"AR ({ar_name.split('/')[-1]})"] = ar_bias

    ar_uniformity = analyzer.compute_uniformity_metric(ar_bias)
    logger.info(f"  AR uniformity (CoV): {ar_uniformity:.4f}")

    # Release AR model
    from src.utils import release_model
    release_model(ar_model)
    del ar_model, ar_tokenizer

    # --- Visualization ---
    plot_position_bias(
        bias_curves,
        title="Position Bias: DiffScore vs AR Model",
        save_path=os.path.join(args.output_dir, "position_bias_curves.pdf"),
    )

    plot_position_bias_distribution(
        bias_curves,
        title="Position Bias Distribution",
        save_path=os.path.join(args.output_dir, "position_bias_distribution.pdf"),
    )

    # Save results
    results = {
        "diffscore": {
            "uniformity_cov": diff_uniformity,
            "mean_std": float(diff_bias[diff_bias > 0].mean()),
        },
        "ar_model": {
            "model": ar_name,
            "uniformity_cov": ar_uniformity,
            "mean_std": float(ar_bias[ar_bias > 0].mean()),
        },
    }

    out_path = os.path.join(args.output_dir, "position_bias_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Experiment 8: Position Bias")
    parser.add_argument("--model_name", type=str, default="GSAI-ML/LLaDA-8B-Instruct")
    parser.add_argument("--ar_model_name", type=str, default="facebook/bart-large-cnn")
    parser.add_argument("--data_path", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="outputs/exp8_position_bias")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--K", type=int, default=10)
    parser.add_argument("--T", type=int, default=10)
    parser.add_argument("--max_samples", type=int, default=100)
    parser.add_argument("--max_positions", type=int, default=200)
    args = parser.parse_args()

    run_experiment(args)
