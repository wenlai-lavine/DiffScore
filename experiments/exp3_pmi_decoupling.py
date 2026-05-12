"""
Experiment 3: PMI Fluency-Relevance Decoupling.

Tests PMI decomposition's ability to separate fluency from relevance
using adversarial test sets:
(a) Fluent-Irrelevant: fluent text unrelated to source
(b) Disfluent-Relevant: noisy text semantically faithful to source

DiffScore_PMI should correctly identify (a) as low-relevance and
(b) as high-relevance, while BARTScore_cond is fooled by (a).
"""

import os
import sys
import json
import logging
import argparse
import numpy as np
from scipy.stats import mannwhitneyu
from tabulate import tabulate

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model import MDLMWrapper
from src.scoring import DiffScorer, PMIScorer
from src.data.summeval import SummEvalDataset
from src.data.adversarial import AdversarialConstructor
from src.baselines.bartscore import BARTScorer
from src.utils import release_model

logger = logging.getLogger(__name__)


def run_experiment(args):
    logging.basicConfig(level=logging.INFO)
    os.makedirs(args.output_dir, exist_ok=True)

    # Load data
    dataset = SummEvalDataset()
    if args.data_path:
        dataset.load_from_file(args.data_path)
    else:
        dataset.load_from_huggingface()

    sources = dataset.get_sources()
    references_raw = dataset.get_references()

    # Flatten multi-references to single strings for adversarial construction
    if references_raw and isinstance(references_raw[0], list):
        references = [ref[0] if ref else "" for ref in references_raw]
    else:
        references = references_raw

    if args.max_samples:
        sources = sources[:args.max_samples]
        references = references[:args.max_samples]

    # Construct adversarial test sets
    logger.info("Constructing adversarial test sets...")
    constructor = AdversarialConstructor()

    fluent_irr = constructor.construct_fluent_irrelevant(sources, references)
    disfluent_rel = constructor.construct_disfluent_relevant(
        sources, references, noise_ratio=args.noise_ratio,
    )

    constructor.save(fluent_irr, os.path.join(args.output_dir, "fluent_irrelevant.jsonl"))
    constructor.save(disfluent_rel, os.path.join(args.output_dir, "disfluent_relevant.jsonl"))

    # Load DiffScore model
    model = MDLMWrapper(
        model_name=args.model_name, device=args.device, dtype=args.dtype,
    )
    scorer = DiffScorer(model, K=args.K, T=args.T, batch_size=args.batch_size)
    pmi_scorer = PMIScorer(scorer)

    prompt_template = dataset.get_prompt_template(args.prompt_template)

    # --- Compute DiffScore on each group ---
    groups = {
        "original": (sources, references),
        "fluent_irrelevant": ([s.source for s in fluent_irr],
                              [s.candidate for s in fluent_irr]),
        "disfluent_relevant": ([s.source for s in disfluent_rel],
                                [s.candidate for s in disfluent_rel]),
    }

    diff_results = {}
    for group_name, (srcs, cands) in groups.items():
        logger.info(f"Computing DiffScore for group: {group_name}")
        pmi_res = pmi_scorer.score_batch(srcs, cands, prompt_template, show_progress=True)

        diff_results[group_name] = {
            "cond": np.array([r.overall for r in pmi_res]),
            "mar": np.array([r.fluency for r in pmi_res]),
            "pmi": np.array([r.relevance for r in pmi_res]),
        }

    # Release DiffScore model before loading BARTScore
    release_model(model)
    del model, scorer, pmi_scorer

    # --- Compute BARTScore on each group ---
    logger.info("Computing BARTScore baselines...")
    bart = BARTScorer(device=args.device)

    bart_results = {}
    for group_name, (srcs, cands) in groups.items():
        logger.info(f"Computing BARTScore for group: {group_name}")
        bart_results[group_name] = {
            "cond": bart.score(srcs, cands, configuration="conditional"),
            "mar": bart.score(srcs, cands, configuration="marginal"),
            "pmi": bart.score(srcs, cands, configuration="pmi"),
        }

    # Release BARTScore model
    release_model(bart)
    del bart

    # --- Analysis ---
    results = {"diffscore": {}, "bartscore": {}}
    for method_name, method_results in [("diffscore", diff_results), ("bartscore", bart_results)]:
        for score_type in ["cond", "mar", "pmi"]:
            stats = {}
            for group in groups:
                vals = method_results[group][score_type]
                stats[group] = {
                    "mean": float(vals.mean()),
                    "std": float(vals.std()),
                }

            # Statistical test: can the metric distinguish groups?
            fi_scores = method_results["fluent_irrelevant"][score_type]
            dr_scores = method_results["disfluent_relevant"][score_type]
            orig_scores = method_results["original"][score_type]

            u_fi_orig, p_fi_orig = mannwhitneyu(fi_scores, orig_scores, alternative="less")
            u_dr_fi, p_dr_fi = mannwhitneyu(dr_scores, fi_scores)

            stats["tests"] = {
                "fluent_irr_vs_original": {
                    "U": float(u_fi_orig), "p_value": float(p_fi_orig),
                },
                "disfluent_rel_vs_fluent_irr": {
                    "U": float(u_dr_fi), "p_value": float(p_dr_fi),
                },
            }
            results[method_name][score_type] = stats

    # Print summary
    headers = ["Method", "Score", "Original", "Fluent-Irr", "Disfluent-Rel"]
    rows = []
    for method in ["diffscore", "bartscore"]:
        for score_type in ["cond", "mar", "pmi"]:
            r = results[method][score_type]
            rows.append([
                method, score_type,
                f"{r['original']['mean']:.4f}±{r['original']['std']:.4f}",
                f"{r['fluent_irrelevant']['mean']:.4f}±{r['fluent_irrelevant']['std']:.4f}",
                f"{r['disfluent_relevant']['mean']:.4f}±{r['disfluent_relevant']['std']:.4f}",
            ])
    print("\nPMI Decoupling Results:")
    print(tabulate(rows, headers=headers, tablefmt="grid"))

    # Save results
    out_path = os.path.join(args.output_dir, "pmi_decoupling_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=float)
    logger.info(f"Results saved to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Experiment 3: PMI Decoupling")
    parser.add_argument("--model_name", type=str, default="GSAI-ML/LLaDA-8B-Instruct")
    parser.add_argument("--data_path", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="outputs/exp3_pmi")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--K", type=int, default=20)
    parser.add_argument("--T", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--noise_ratio", type=float, default=0.3)
    parser.add_argument("--prompt_template", type=str, default="default")
    args = parser.parse_args()

    run_experiment(args)
