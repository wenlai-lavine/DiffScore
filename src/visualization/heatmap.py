"""
Heatmap visualizations for DiffScore.

- Token-level quality heatmap (Experiment 7)
- Dimension x Timestep correlation heatmap (Experiment 2)
- Multi-benchmark comparison heatmap
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
from typing import Dict, List, Optional
import os
import logging

logger = logging.getLogger(__name__)


def plot_token_heatmap(
    tokens: List[str],
    scores: np.ndarray,
    title: str = "Token-Level Quality Scores",
    save_path: Optional[str] = None,
    figsize: Optional[tuple] = None,
    cmap: str = "RdYlGn",
    max_tokens_per_row: int = 20,
):
    """Visualize per-token quality scores as a colored text heatmap.

    High scores (green) indicate well-predicted tokens.
    Low scores (red) indicate potentially problematic tokens.
    """
    n_tokens = len(tokens)
    n_rows = (n_tokens + max_tokens_per_row - 1) // max_tokens_per_row

    if figsize is None:
        figsize = (min(20, max_tokens_per_row * 0.8), max(2, n_rows * 1.2))

    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlim(0, max_tokens_per_row)
    ax.set_ylim(-n_rows, 0.5)
    ax.axis("off")
    ax.set_title(title, fontsize=14, fontweight="bold", pad=10)

    # Normalize scores for colormap
    norm = plt.Normalize(vmin=scores.min(), vmax=scores.max())
    colormap = plt.cm.get_cmap(cmap)

    for i, (token, score) in enumerate(zip(tokens, scores)):
        row = i // max_tokens_per_row
        col = i % max_tokens_per_row

        color = colormap(norm(score))
        ax.text(
            col + 0.5, -row, token.replace("▁", ""),
            ha="center", va="center",
            fontsize=9,
            bbox=dict(
                boxstyle="round,pad=0.2",
                facecolor=color,
                edgecolor="gray",
                alpha=0.8,
            ),
        )

    # Colorbar
    sm = plt.cm.ScalarMappable(cmap=colormap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, shrink=0.6, pad=0.02)
    cbar.set_label("Log-probability Score", fontsize=10)

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved token heatmap to {save_path}")
    plt.close()


def plot_timestep_dimension_heatmap(
    correlation_matrix: Dict[str, Dict[float, float]],
    title: str = "Dimension × Timestep Correlation",
    save_path: Optional[str] = None,
    figsize: tuple = (12, 5),
    cmap: str = "YlOrRd",
):
    """Plot dimension × timestep correlation heatmap (Experiment 2).

    Shows which evaluation dimensions correlate best at which timesteps.
    """
    dimensions = list(correlation_matrix.keys())
    timesteps = sorted(list(next(iter(correlation_matrix.values())).keys()))

    matrix = np.zeros((len(dimensions), len(timesteps)))
    for i, dim in enumerate(dimensions):
        for j, t in enumerate(timesteps):
            matrix[i, j] = correlation_matrix[dim].get(t, 0.0)

    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        matrix, ax=ax,
        xticklabels=[f"t={t:.1f}" for t in timesteps],
        yticklabels=[d.capitalize() for d in dimensions],
        annot=True, fmt=".3f",
        cmap=cmap,
        vmin=0, vmax=max(0.5, matrix.max()),
        linewidths=0.5,
    )
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel("Timestep (mask ratio)", fontsize=12)
    ax.set_ylabel("Evaluation Dimension", fontsize=12)

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved dimension-timestep heatmap to {save_path}")
    plt.close()


def plot_benchmark_comparison_heatmap(
    results: Dict[str, Dict[str, float]],
    title: str = "Multi-Benchmark Method Comparison (Spearman ρ)",
    save_path: Optional[str] = None,
    figsize: tuple = (14, 8),
    cmap: str = "YlOrRd",
    methods_order: Optional[List[str]] = None,
    benchmarks_order: Optional[List[str]] = None,
):
    """Plot a method x benchmark comparison heatmap.

    Args:
        results: {method_name: {benchmark_dim: spearman_rho}}
        methods_order: optional ordering for methods (y-axis)
        benchmarks_order: optional ordering for benchmarks (x-axis)
    """
    if methods_order is None:
        methods_order = sorted(results.keys())
    if benchmarks_order is None:
        all_benchmarks = set()
        for m in results.values():
            all_benchmarks.update(m.keys())
        benchmarks_order = sorted(all_benchmarks)

    matrix = np.zeros((len(methods_order), len(benchmarks_order)))
    for i, method in enumerate(methods_order):
        for j, bench in enumerate(benchmarks_order):
            matrix[i, j] = results.get(method, {}).get(bench, np.nan)

    fig, ax = plt.subplots(figsize=figsize)
    mask = np.isnan(matrix)

    sns.heatmap(
        matrix, ax=ax, mask=mask,
        xticklabels=benchmarks_order,
        yticklabels=methods_order,
        annot=True, fmt=".3f",
        cmap=cmap,
        vmin=0, vmax=max(0.6, np.nanmax(matrix)),
        linewidths=0.5,
    )
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel("Benchmark - Dimension", fontsize=11)
    ax.set_ylabel("Method", fontsize=11)
    plt.xticks(rotation=45, ha="right", fontsize=9)
    plt.yticks(fontsize=9)

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved benchmark comparison heatmap to {save_path}")
    plt.close()
