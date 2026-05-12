"""
Experiment 4: Bidirectional Reasoning Consistency Test.

Tests whether DiffScore produces more consistent scores for
semantically equivalent forward/reverse formulations compared
to AR models (BARTScore, GPTScore).

DirConsist = 1 - |Score(fwd) - Score(rev)| / (|Score(fwd)| + |Score(rev)|)
"""

import os
import sys
import json
import logging
import argparse
import numpy as np
from tabulate import tabulate

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model import MDLMWrapper
from src.scoring import DiffScorer
from src.data.adversarial import ReversalCurseConstructor
from src.analysis.direction_consistency import DirectionConsistencyAnalyzer

logger = logging.getLogger(__name__)


def run_experiment(args):
    logging.basicConfig(level=logging.INFO)
    os.makedirs(args.output_dir, exist_ok=True)

    # Construct or load reversal pairs
    constructor = ReversalCurseConstructor()
    pairs = constructor.construct_pairs(
        data_path=args.reversal_data_path,
        n_synthetic=args.n_pairs,
    )
    logger.info(f"Using {len(pairs)} forward/reverse pairs")

    constructor.save_pairs(pairs, os.path.join(args.output_dir, "reversal_pairs.jsonl"))

    analyzer = DirectionConsistencyAnalyzer()
    results = {}

    # --- DiffScore ---
    logger.info("Evaluating DiffScore direction consistency...")
    model = MDLMWrapper(
        model_name=args.model_name, device=args.device, dtype=args.dtype,
    )
    scorer = DiffScorer(model, K=args.K, T=args.T, batch_size=args.batch_size)

    diff_result = analyzer.evaluate_diffscore(scorer, pairs)
    results["diffscore"] = {
        "mean_consistency": diff_result.mean_consistency,
        "std_consistency": diff_result.std_consistency,
        "forward_mean": float(diff_result.forward_scores.mean()),
        "reverse_mean": float(diff_result.reverse_scores.mean()),
        "correlation": float(np.corrcoef(
            diff_result.forward_scores, diff_result.reverse_scores
        )[0, 1]),
    }
    logger.info(f"  DiffScore consistency: {diff_result.mean_consistency:.4f} ± {diff_result.std_consistency:.4f}")

    # Clean up DiffScore model to free memory
    del model, scorer
    import torch
    torch.cuda.empty_cache()

    # --- BARTScore (AR baseline) ---
    logger.info("Evaluating BARTScore direction consistency...")
    from transformers import BartForConditionalGeneration, BartTokenizer
    bart_name = "facebook/bart-large-cnn"
    bart_tokenizer = BartTokenizer.from_pretrained(bart_name)
    bart_model = BartForConditionalGeneration.from_pretrained(bart_name).to(args.device)
    bart_model.eval()

    import torch.nn.functional as F
    forward_scores = []
    reverse_scores = []
    for fwd, rev in pairs:
        for text, score_list in [(fwd, forward_scores), (rev, reverse_scores)]:
            enc = bart_tokenizer.encode(text, add_special_tokens=True)
            input_ids = torch.tensor([enc], dtype=torch.long).to(args.device)

            with torch.no_grad():
                # Decoder-only scoring
                output = bart_model(input_ids=input_ids, decoder_input_ids=input_ids)
                logits = output.logits
                log_probs = F.log_softmax(logits, dim=-1)

                token_lps = []
                for pos in range(1, len(enc)):
                    lp = log_probs[0, pos - 1, enc[pos]].item()
                    token_lps.append(lp)
                score_list.append(np.mean(token_lps) if token_lps else 0.0)

    # Release BART model
    from src.utils import release_model
    release_model(bart_model)
    del bart_model, bart_tokenizer

    bart_consistency = analyzer.compute_consistency(
        np.array(forward_scores), np.array(reverse_scores)
    )
    results["bartscore"] = {
        "mean_consistency": bart_consistency.mean_consistency,
        "std_consistency": bart_consistency.std_consistency,
        "forward_mean": float(bart_consistency.forward_scores.mean()),
        "reverse_mean": float(bart_consistency.reverse_scores.mean()),
        "correlation": float(np.corrcoef(
            bart_consistency.forward_scores, bart_consistency.reverse_scores
        )[0, 1]),
    }
    logger.info(f"  BARTScore consistency: {bart_consistency.mean_consistency:.4f} ± {bart_consistency.std_consistency:.4f}")

    # Print comparison
    headers = ["Model", "Mean Consistency", "Std", "Fwd-Rev Correlation"]
    rows = []
    for name, r in results.items():
        rows.append([
            name,
            f"{r['mean_consistency']:.4f}",
            f"{r['std_consistency']:.4f}",
            f"{r['correlation']:.4f}",
        ])
    print("\nDirection Consistency Results:")
    print(tabulate(rows, headers=headers, tablefmt="grid"))

    # Save results
    out_path = os.path.join(args.output_dir, "direction_consistency_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=float)
    logger.info(f"Results saved to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Experiment 4: Direction Consistency")
    parser.add_argument("--model_name", type=str, default="GSAI-ML/LLaDA-8B-Instruct")
    parser.add_argument("--output_dir", type=str, default="outputs/exp4_direction")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--K", type=int, default=50)
    parser.add_argument("--T", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--n_pairs", type=int, default=100)
    parser.add_argument("--reversal_data_path", type=str, default=None)
    args = parser.parse_args()

    run_experiment(args)
