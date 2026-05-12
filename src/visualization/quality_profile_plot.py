"""
Quality profile curve visualization (Experiment 2 & 7).

Plots S(t) curves for comparing high-quality vs low-quality texts.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional
import os
import logging

logger = logging.getLogger(__name__)


def plot_quality_profiles(
    profiles: Dict[str, Dict[float, float]],
    title: str = "Multi-Timestep Quality Profiles",
    save_path: Optional[str] = None,
    figsize: tuple = (10, 6),
    ylabel: str = "Score S(t)",
):
    """Plot quality profile curves for multiple texts/groups.

    Args:
        profiles: {label: {timestep: score}}
    """
    fig, ax = plt.subplots(figsize=figsize)

    colors = plt.cm.tab10(np.linspace(0, 1, len(profiles)))
    for (label, profile), color in zip(profiles.items(), colors):
        timesteps = sorted(profile.keys())
        scores = [profile[t] for t in timesteps]
        ax.plot(timesteps, scores, "o-", label=label, color=color,
                linewidth=2, markersize=6)

    ax.set_xlabel("Timestep t (mask ratio)", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved quality profiles to {save_path}")
    plt.close()


def plot_convergence_curve(
    K_values: List[int],
    correlations: Dict[str, List[float]],
    title: str = "MC Sampling Convergence",
    save_path: Optional[str] = None,
    figsize: tuple = (8, 5),
):
    """Plot Spearman rho vs. K to verify MC convergence (Sanity Check).

    Args:
        K_values: list of K values tested
        correlations: {dimension: [rho_at_K1, rho_at_K2, ...]}
    """
    fig, ax = plt.subplots(figsize=figsize)

    for dim_name, rhos in correlations.items():
        ax.plot(K_values, rhos, "o-", label=dim_name, linewidth=2, markersize=6)

    ax.set_xlabel("Number of MC samples K", fontsize=12)
    ax.set_ylabel("Spearman ρ", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved convergence curve to {save_path}")
    plt.close()


def plot_pareto_frontier(
    methods: List[str],
    performance: List[float],
    efficiency: List[float],
    title: str = "Performance-Efficiency Pareto Frontier",
    save_path: Optional[str] = None,
    figsize: tuple = (8, 6),
    perf_label: str = "Spearman ρ",
    eff_label: str = "Time per sample (s)",
):
    """Plot performance vs efficiency for Experiment 6."""
    fig, ax = plt.subplots(figsize=figsize)

    ax.scatter(efficiency, performance, s=100, zorder=5)
    for i, method in enumerate(methods):
        ax.annotate(
            method, (efficiency[i], performance[i]),
            textcoords="offset points", xytext=(5, 5), fontsize=9,
        )

    # Draw Pareto frontier
    points = sorted(zip(efficiency, performance), key=lambda x: x[0])
    pareto_x, pareto_y = [points[0][0]], [points[0][1]]
    best_y = points[0][1]
    for x, y in points[1:]:
        if y > best_y:
            pareto_x.append(x)
            pareto_y.append(y)
            best_y = y
    ax.plot(pareto_x, pareto_y, "r--", alpha=0.5, linewidth=1.5,
            label="Pareto frontier")

    ax.set_xlabel(eff_label, fontsize=12)
    ax.set_ylabel(perf_label, fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved Pareto plot to {save_path}")
    plt.close()
