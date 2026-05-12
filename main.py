"""
DiffScore: Main entry point for running experiments.

Usage:
    python main.py --experiment <exp_id> [experiment-specific args]

Experiments:
    0: Sanity Check (pre-validation)
    1: Meta-Evaluation (main results)
    2: Quality Profile Analysis
    3: PMI Fluency-Relevance Decoupling
    4: Bidirectional Reasoning Consistency
    5: Ablation Study
    6: Efficiency Analysis
    7: Interpretability Case Study
    8: Position Bias Analysis
"""

import argparse
import subprocess
import sys
import os


EXPERIMENTS = {
    0: ("experiments/exp0_sanity_check.py", "Sanity Check"),
    1: ("experiments/exp1_meta_evaluation.py", "Meta-Evaluation"),
    2: ("experiments/exp2_quality_profile.py", "Quality Profile"),
    3: ("experiments/exp3_pmi_decoupling.py", "PMI Decoupling"),
    4: ("experiments/exp4_direction_consistency.py", "Direction Consistency"),
    5: ("experiments/exp5_ablation.py", "Ablation Study"),
    6: ("experiments/exp6_efficiency.py", "Efficiency Analysis"),
    7: ("experiments/exp7_case_study.py", "Case Study"),
    8: ("experiments/exp8_position_bias.py", "Position Bias"),
}


def main():
    parser = argparse.ArgumentParser(
        description="DiffScore: Multi-granularity Text Evaluation via Masked Diffusion LLMs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Experiments:\n" + "\n".join([f"  {k}: {v[1]}" for k, v in EXPERIMENTS.items()]),
    )
    parser.add_argument(
        "--experiment", type=int, required=True,
        choices=list(EXPERIMENTS.keys()),
        help="Experiment ID to run (0-8)",
    )

    args, remaining = parser.parse_known_args()
    script_path, exp_name = EXPERIMENTS[args.experiment]

    print(f"\n{'='*60}")
    print(f"Running Experiment {args.experiment}: {exp_name}")
    print(f"{'='*60}\n")

    cmd = [sys.executable, script_path] + remaining
    result = subprocess.run(cmd, cwd=os.path.dirname(os.path.abspath(__file__)))
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
