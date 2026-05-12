"""
Position bias visualization (Experiment 8).

Plots per-position scoring standard deviation for DiffScore vs AR models.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Optional
import os
import logging

logger = logging.getLogger(__name__)


def plot_position_bias(
    bias_curves: Dict[str, np.ndarray],
    title: str = "Position Bias: Scoring Std by Token Position",
    save_path: Optional[str] = None,
    figsize: tuple = (12, 5),
    smooth_window: int = 5,
):
    """Plot position bias curves for multiple models.

    Args:
        bias_curves: {model_name: (max_pos,) std values}
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Left: raw curves
    ax = axes[0]
    for name, curve in bias_curves.items():
        valid = curve > 0
        positions = np.arange(len(curve))[valid]
        values = curve[valid]
        ax.plot(positions, values, alpha=0.6, linewidth=1, label=name)

    ax.set_xlabel("Token Position", fontsize=11)
    ax.set_ylabel("Std of Log-prob Score", fontsize=11)
    ax.set_title("Raw Position Bias", fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Right: smoothed curves
    ax = axes[1]
    for name, curve in bias_curves.items():
        valid = curve > 0
        positions = np.arange(len(curve))[valid]
        values = curve[valid]

        if len(values) > smooth_window:
            kernel = np.ones(smooth_window) / smooth_window
            smoothed = np.convolve(values, kernel, mode="valid")
            smooth_pos = positions[:len(smoothed)]
            ax.plot(smooth_pos, smoothed, linewidth=2, label=name)
        else:
            ax.plot(positions, values, linewidth=2, label=name)

    ax.set_xlabel("Token Position", fontsize=11)
    ax.set_ylabel("Smoothed Std", fontsize=11)
    ax.set_title(f"Smoothed (window={smooth_window})", fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.suptitle(title, fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved position bias plot to {save_path}")
    plt.close()


def plot_position_bias_distribution(
    bias_curves: Dict[str, np.ndarray],
    title: str = "Distribution of Position Bias",
    save_path: Optional[str] = None,
    figsize: tuple = (8, 5),
):
    """Box/violin plot comparing position bias distributions."""
    import seaborn as sns

    fig, ax = plt.subplots(figsize=figsize)

    data = []
    labels = []
    for name, curve in bias_curves.items():
        valid = curve[curve > 0]
        data.extend(valid.tolist())
        labels.extend([name] * len(valid))

    import pandas as pd
    df = pd.DataFrame({"Std of Score": data, "Model": labels})
    sns.violinplot(data=df, x="Model", y="Std of Score", ax=ax, inner="box")

    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_ylabel("Std of Log-prob Score", fontsize=12)

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved position bias distribution to {save_path}")
    plt.close()
