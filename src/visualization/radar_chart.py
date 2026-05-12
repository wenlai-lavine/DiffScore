"""
Radar chart showing DiffScore performance across 16 BartScore-aligned benchmarks.
Designed for placement between Abstract and Introduction in the paper.
"""

import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUTS = PROJECT_ROOT / "outputs" / "llada"


def load_json(path):
    with open(path) as f:
        return json.load(f)


def extract_benchmark_scores():
    """Extract single scalar per benchmark for each method.
    
    Returns dict: {method_name: {benchmark_name: scalar_value}}
    """
    baseline_wmt = load_json(OUTPUTS / "baseline" / "exp1_meta_eval_wmt" / "meta_evaluation_results.json")
    baseline_sum = load_json(OUTPUTS / "baseline" / "exp1_meta_eval_sum" / "meta_evaluation_results.json")
    baseline_d2t = load_json(OUTPUTS / "baseline" / "exp1_meta_eval_d2t" / "meta_evaluation_results.json")

    sft_wmt = load_json(OUTPUTS / "sft_all" / "sft_epoch_3" / "exp1_meta_eval_wmt" / "meta_evaluation_results.json")
    sft_sum = load_json(OUTPUTS / "sft_all" / "sft_epoch_3" / "exp1_meta_eval_sum" / "meta_evaluation_results.json")
    sft_d2t = load_json(OUTPUTS / "sft_all" / "sft_epoch_3" / "exp1_meta_eval_d2t" / "meta_evaluation_results.json")

    wmt_langs = ["de-en", "fi-en", "gu-en", "kk-en", "lt-en", "ru-en", "zh-en"]

    methods = {}

    # --- DiffScore-Zero: use best config per task ---
    ds_zero = {}
    for lang in wmt_langs:
        key = f"wmt_{lang}"
        candidates = ["diffscore_cond", "diffscore_bi_a5", "diffscore_bi_a7"]
        ds_zero[lang] = max(baseline_wmt[key][c]["kendall_tau"] for c in candidates)

    for name, dims in [("Newsroom", ["coherence", "fluency", "informativeness", "relevance"]),
                        ("SummEval", ["coherence", "consistency", "fluency", "relevance"])]:
        key = f"sum_{name}"
        vals = []
        for dim in dims:
            candidates = ["diffscore_cond", "diffscore_bi_a5", "diffscore_bi_a7"]
            vals.append(max(baseline_sum[key][dim][c]["spearman"] for c in candidates))
        ds_zero[name] = np.mean(vals)

    ds_zero["REALSumm"] = baseline_sum["sum_REALSumm"]["litepyramid_recall"]["diffscore_rev"]["spearman"]
    ds_zero["Rank19"] = baseline_sum["sum_Rank19"]["fact"]["diffscore_bi_a7"]["accuracy"]
    ds_zero["QAGS_CNN"] = baseline_sum["sum_QAGS_CNN"]["fact"]["diffscore_cond"]["pearson"]
    ds_zero["QAGS_XSUM"] = baseline_sum["sum_QAGS_XSUM"]["fact"]["diffscore_pmi"]["pearson"]

    for name in ["BAGEL", "SFHOT", "SFRES"]:
        key = f"d2t_{name}"
        vals = []
        for dim in ["informativeness", "naturalness", "quality"]:
            candidates = ["diffscore_cond_src", "diffscore_cond_ref", "diffscore_bi_a7"]
            vals.append(max(baseline_d2t[key][dim][c]["spearman"] for c in candidates))
        ds_zero[name] = np.mean(vals)

    methods["DiffScore-Zero"] = ds_zero

    # --- DiffScore-FT (sft_all epoch 3) ---
    ds_ft = {}
    for lang in wmt_langs:
        key = f"wmt_{lang}"
        candidates = ["diffscore_ft_cond", "diffscore_ft_bi_a5", "diffscore_ft_bi_a7"]
        ds_ft[lang] = max(sft_wmt[key][c]["kendall_tau"] for c in candidates)

    for name, dims in [("Newsroom", ["coherence", "fluency", "informativeness", "relevance"]),
                        ("SummEval", ["coherence", "consistency", "fluency", "relevance"])]:
        key = f"sum_{name}"
        vals = []
        for dim in dims:
            candidates = ["diffscore_ft_cond", "diffscore_ft_bi_a5", "diffscore_ft_bi_a7"]
            vals.append(max(sft_sum[key][dim][c]["spearman"] for c in candidates))
        ds_ft[name] = np.mean(vals)

    ds_ft["REALSumm"] = sft_sum["sum_REALSumm"]["litepyramid_recall"]["diffscore_ft_rev"]["spearman"]
    ds_ft["Rank19"] = sft_sum["sum_Rank19"]["fact"]["diffscore_ft_cond"]["accuracy"]
    ds_ft["QAGS_CNN"] = sft_sum["sum_QAGS_CNN"]["fact"]["diffscore_ft_cond"]["pearson"]
    ds_ft["QAGS_XSUM"] = sft_sum["sum_QAGS_XSUM"]["fact"]["diffscore_ft_pmi"]["pearson"]

    for name in ["BAGEL", "SFHOT", "SFRES"]:
        key = f"d2t_{name}"
        vals = []
        for dim in ["informativeness", "naturalness", "quality"]:
            candidates = ["diffscore_ft_cond_src", "diffscore_ft_cond_ref", "diffscore_ft_bi_a7"]
            vals.append(max(sft_d2t[key][dim][c]["spearman"] for c in candidates))
        ds_ft[name] = np.mean(vals)

    methods["DiffScore-FT"] = ds_ft

    # --- BARTScore (best variant per benchmark) ---
    bs = {}
    for lang in wmt_langs:
        key = f"wmt_{lang}"
        bs[lang] = baseline_wmt[key]["bartscore_avg_f"]["kendall_tau"]

    for name, dims in [("Newsroom", ["coherence", "fluency", "informativeness", "relevance"]),
                        ("SummEval", ["coherence", "consistency", "fluency", "relevance"])]:
        key = f"sum_{name}"
        vals = [baseline_sum[key][dim]["bartscore_src_hypo"]["spearman"] for dim in dims]
        bs[name] = np.mean(vals)

    bs["REALSumm"] = baseline_sum["sum_REALSumm"]["litepyramid_recall"]["bartscore_hypo_ref"]["spearman"]
    bs["Rank19"] = baseline_sum["sum_Rank19"]["fact"]["bartscore_src_hypo"]["accuracy"]
    bs["QAGS_CNN"] = baseline_sum["sum_QAGS_CNN"]["fact"]["bartscore_src_hypo"]["pearson"]
    bs["QAGS_XSUM"] = baseline_sum["sum_QAGS_XSUM"]["fact"]["bartscore_src_hypo"]["pearson"]

    for name in ["BAGEL", "SFHOT", "SFRES"]:
        key = f"d2t_{name}"
        vals = [baseline_d2t[key][dim]["bartscore_ref_hypo"]["spearman"] for dim in ["informativeness", "naturalness", "quality"]]
        bs[name] = np.mean(vals)

    methods["BARTScore"] = bs

    # --- BERTScore ---
    bert = {}
    for lang in wmt_langs:
        key = f"wmt_{lang}"
        bert[lang] = baseline_wmt[key]["bertscore_f1"]["kendall_tau"]

    for name, dims in [("Newsroom", ["coherence", "fluency", "informativeness", "relevance"]),
                        ("SummEval", ["coherence", "consistency", "fluency", "relevance"])]:
        key = f"sum_{name}"
        vals = [baseline_sum[key][dim]["bertscore_f1"]["spearman"] for dim in dims]
        bert[name] = np.mean(vals)

    bert["REALSumm"] = baseline_sum["sum_REALSumm"]["litepyramid_recall"]["bertscore_f1"]["spearman"]
    bert["Rank19"] = baseline_sum["sum_Rank19"]["fact"]["bertscore_f1"]["accuracy"]
    bert["QAGS_CNN"] = baseline_sum["sum_QAGS_CNN"]["fact"]["bertscore_f1"]["pearson"]
    bert["QAGS_XSUM"] = baseline_sum["sum_QAGS_XSUM"]["fact"]["bertscore_f1"]["pearson"]

    for name in ["BAGEL", "SFHOT", "SFRES"]:
        key = f"d2t_{name}"
        vals = [baseline_d2t[key][dim]["bertscore_f1"]["spearman"] for dim in ["informativeness", "naturalness", "quality"]]
        bert[name] = np.mean(vals)

    methods["BERTScore"] = bert

    return methods


def plot_radar(methods, output_path, benchmarks=None):
    """Draw a radar chart comparing methods across benchmarks."""
    if benchmarks is None:
        benchmarks = [
            "de-en", "fi-en", "gu-en", "kk-en", "lt-en", "ru-en", "zh-en",
            "Newsroom", "SummEval", "REALSumm", "Rank19", "QAGS_CNN",
            "QAGS_XSUM", "BAGEL", "SFHOT", "SFRES",
        ]

    labels = [
        "de-en", "fi-en", "gu-en", "kk-en", "lt-en", "ru-en", "zh-en",
        "NR", "SE", "RS", "R19", "Q-CNN",
        "Q-XS", "BAG", "SFH", "SFR",
    ]

    N = len(benchmarks)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')

    style_map = {
        "DiffScore-Zero": {"color": "#2196F3", "linestyle": "-", "linewidth": 2.2, "marker": "o", "markersize": 5, "zorder": 4},
        "DiffScore-FT":   {"color": "#E53935", "linestyle": "-", "linewidth": 2.2, "marker": "s", "markersize": 5, "zorder": 5},
        "BARTScore":      {"color": "#757575", "linestyle": "--", "linewidth": 1.8, "marker": "^", "markersize": 5, "zorder": 3},
        "BERTScore":      {"color": "#43A047", "linestyle": ":", "linewidth": 1.5, "marker": "d", "markersize": 4, "zorder": 2},
    }

    method_order = ["BERTScore", "BARTScore", "DiffScore-Zero", "DiffScore-FT"]

    for method_name in method_order:
        data = methods[method_name]
        values = [data[b] for b in benchmarks]
        values += values[:1]
        style = style_map[method_name]
        ax.plot(angles, values, label=method_name, **style)
        ax.fill(angles, values, alpha=0.04, color=style["color"])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=9, fontweight='bold')

    ax.set_ylim(0, 0.85)
    ax.set_yticks([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
    ax.set_yticklabels(["0.1", "0.2", "0.3", "0.4", "0.5", "0.6", "0.7", "0.8"],
                       fontsize=7, color="grey")
    ax.yaxis.set_tick_params(labelsize=7)

    for i, (bench, label) in enumerate(zip(benchmarks, labels)):
        angle_rad = angles[i]
        for method_name in ["DiffScore-FT"]:
            val = methods[method_name][bench]

    # Sector background shading
    sector_info = [
        (0, 7, "#E3F2FD", "WMT"),
        (7, 12, "#FFF3E0", "SUM"),
        (12, 13, "#FFF3E0", ""),
        (13, 16, "#E8F5E9", "D2T"),
    ]
    for start, end, color, sector_label in sector_info:
        theta1 = angles[start]
        theta2 = angles[end] if end < N else angles[0]
        if end >= N:
            theta2 = 2 * np.pi
        sector_angles = np.linspace(theta1, theta2, 50)
        r_outer = 0.85
        for a in sector_angles:
            ax.plot([a, a], [0, r_outer], color=color, alpha=0.15, linewidth=0.3)

    # Sector labels
    sector_label_info = [
        (3.5, "WMT (7 lang-pairs)"),
        (9.5, "SUM (6 datasets)"),
        (14.5, "D2T (3 datasets)"),
    ]
    for midpoint_idx, slabel in sector_label_info:
        mid_angle = angles[int(midpoint_idx)] if midpoint_idx == int(midpoint_idx) else \
            (angles[int(midpoint_idx)] + angles[int(midpoint_idx) + 1]) / 2
        ax.text(mid_angle, 0.92, slabel, ha='center', va='center',
                fontsize=7.5, color='#555', fontstyle='italic')

    ax.legend(loc='upper right', bbox_to_anchor=(1.32, 1.12),
              fontsize=10, frameon=True, fancybox=True, shadow=False,
              edgecolor='#ccc')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Radar chart saved to {output_path}")


def compute_summary_stats(methods):
    """Print summary statistics (wins, average correlation)."""
    benchmarks = list(methods["DiffScore-Zero"].keys())

    print("\n=== Summary Statistics ===")
    for mname, mdata in methods.items():
        avg = np.mean(list(mdata.values()))
        print(f"{mname:20s}  Avg: {avg:.3f}")

    wins = {m: 0 for m in methods}
    for b in benchmarks:
        best_method = max(methods, key=lambda m: methods[m][b])
        wins[best_method] += 1

    print("\nWins across 16 benchmarks:")
    for m, w in wins.items():
        print(f"  {m:20s}: {w}/16")


if __name__ == "__main__":
    methods = extract_benchmark_scores()
    out_dir = PROJECT_ROOT / "outputs" / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_radar(methods, out_dir / "radar_16benchmarks.pdf")
    compute_summary_stats(methods)
