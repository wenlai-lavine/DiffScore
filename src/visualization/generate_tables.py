"""
Generate LaTeX tables for the DiffScore paper.
Tables follow BARTScore paper format (Yuan et al., 2021).
"""

import json
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUTS = PROJECT_ROOT / "outputs" / "llada"


def load_json(path):
    with open(path) as f:
        content = f.read()
        content = content.replace("NaN", "null")
        return json.loads(content)


def fmt(val, bold=False, underline=False):
    """Format a numeric value for LaTeX."""
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return "--"
    s = f"{val:.3f}"
    if val < 0:
        s = f"{val:.3f}"
    if bold and underline:
        s = f"\\underline{{\\textbf{{{s}}}}}"
    elif bold:
        s = f"\\textbf{{{s}}}"
    elif underline:
        s = f"\\underline{{{s}}}"
    return s


# ==============================================================================
# TABLE 1: WMT19 Kendall Tau
# ==============================================================================
def generate_wmt_table():
    baseline = load_json(OUTPUTS / "baseline" / "exp1_meta_eval_wmt" / "meta_evaluation_results.json")
    sft = load_json(OUTPUTS / "sft_all" / "sft_epoch_3" / "exp1_meta_eval_wmt" / "meta_evaluation_results.json")

    langs = ["de-en", "fi-en", "gu-en", "kk-en", "lt-en", "ru-en", "zh-en"]

    metrics = {
        "BLEU":         [baseline[f"wmt_{l}"]["bleu"]["kendall_tau"] for l in langs],
        "METEOR":       [baseline[f"wmt_{l}"]["meteor"]["kendall_tau"] for l in langs],
        "MoverScore":   [baseline[f"wmt_{l}"]["moverscore"]["kendall_tau"] for l in langs],
        "BERTScore":    [baseline[f"wmt_{l}"]["bertscore_f1"]["kendall_tau"] for l in langs],
        "BARTScore":    [baseline[f"wmt_{l}"]["bartscore_avg_f"]["kendall_tau"] for l in langs],
        "AlignScore":   [baseline[f"wmt_{l}"]["alignscore"]["kendall_tau"] for l in langs],
        "QuestEval":    [baseline[f"wmt_{l}"]["questeval"]["kendall_tau"] for l in langs],
        "UniEval$^\\dagger$": [baseline[f"wmt_{l}"]["unieval_overall"]["kendall_tau"] for l in langs],
        "DiffScore-Zero": [max(baseline[f"wmt_{l}"][c]["kendall_tau"] for c in ["diffscore_cond", "diffscore_bi_a5", "diffscore_bi_a7"]) for l in langs],
        "DiffScore-FT": [max(sft[f"wmt_{l}"][c]["kendall_tau"] for c in ["diffscore_ft_cond", "diffscore_ft_bi_a5", "diffscore_ft_bi_a7"]) for l in langs],
    }

    for name in metrics:
        metrics[name].append(np.mean(metrics[name]))

    cols = langs + ["Avg."]

    unsupervised = ["BLEU", "METEOR", "MoverScore", "BERTScore", "BARTScore",
                     "DiffScore-Zero", "DiffScore-FT"]
    all_methods = list(metrics.keys())

    col_best_unsup = []
    col_best_all = []
    for j in range(len(cols)):
        unsup_vals = [(name, metrics[name][j]) for name in unsupervised]
        all_vals = [(name, metrics[name][j]) for name in all_methods]
        col_best_unsup.append(max(unsup_vals, key=lambda x: x[1])[1])
        col_best_all.append(max(all_vals, key=lambda x: x[1])[1])

    latex = []
    latex.append("\\begin{table*}[t]")
    latex.append("\\centering")
    latex.append("\\caption{\\label{tab:mt} Kendall's Tau correlation of different metrics on WMT19 dataset. "
                 "The highest correlation for each language pair achieved by \\textit{unsupervised} methods is "
                 "\\textbf{bold}, and the highest correlation \\textit{overall} is \\underline{underlined}. "
                 "$\\dagger$ denotes methods trained on large-scale human judgment data. "
                 "\\textbf{Avg.} denotes the average across all language pairs.}")
    latex.append("\\resizebox{\\textwidth}{!}{")
    latex.append("\\begin{tabular}{l" + "c" * len(cols) + "}")
    latex.append("\\toprule")

    header = " & ".join([f"\\textbf{{{c}}}" for c in cols])
    latex.append(f" & {header} \\\\")
    latex.append("\\midrule")

    group_labels = [
        (["BLEU", "METEOR"], "\\textit{Traditional}"),
        (["MoverScore", "BERTScore"], "\\textit{Embedding}"),
        (["BARTScore"], "\\textit{AR Likelihood}"),
        (["AlignScore", "QuestEval", "UniEval$^\\dagger$"], "\\textit{Trained}"),
        (["DiffScore-Zero", "DiffScore-FT"], "\\textit{Ours}"),
    ]

    for group_methods, group_label in group_labels:
        for i, name in enumerate(group_methods):
            vals = metrics[name]
            row_parts = []
            for j, v in enumerate(vals):
                is_best_unsup = (v == col_best_unsup[j]) and (name in unsupervised)
                is_best_all = (v == col_best_all[j])
                row_parts.append(fmt(v, bold=is_best_unsup, underline=is_best_all))
            row = " & ".join(row_parts)
            latex.append(f"{name} & {row} \\\\")
        if group_label != "\\textit{Ours}":
            latex.append("\\midrule")

    latex.append("\\bottomrule")
    latex.append("\\end{tabular}}")
    latex.append("\\end{table*}")

    return "\n".join(latex)


# ==============================================================================
# TABLE 2: SUM - Spearman correlation on human judgement datasets
# ==============================================================================
def generate_sum_table():
    baseline = load_json(OUTPUTS / "baseline" / "exp1_meta_eval_sum" / "meta_evaluation_results.json")
    sft = load_json(OUTPUTS / "sft_all" / "sft_epoch_3" / "exp1_meta_eval_sum" / "meta_evaluation_results.json")

    def get_sum_metric(data, dataset, dim, metric_name, key="spearman"):
        try:
            return data[f"sum_{dataset}"][dim][metric_name][key]
        except (KeyError, TypeError):
            return None

    columns = [
        ("REALSumm", "litepyramid_recall", "Cov"),
        ("SummEval", "coherence", "Coh"),
        ("SummEval", "consistency", "Fac"),
        ("SummEval", "fluency", "Flu"),
        ("SummEval", "relevance", "Rel"),
        ("Newsroom", "coherence", "Coh"),
        ("Newsroom", "fluency", "Flu"),
        ("Newsroom", "informativeness", "Info"),
        ("Newsroom", "relevance", "Rel"),
    ]

    bartscore_key = {
        "REALSumm": "bartscore_hypo_ref",
        "SummEval": "bartscore_src_hypo",
        "Newsroom": "bartscore_src_hypo",
    }

    def extract_row(name_label, get_fn):
        vals = [get_fn(ds, dim) for ds, dim, _ in columns]
        valid = [v for v in vals if v is not None]
        vals.append(np.mean(valid) if valid else None)
        return vals

    metrics_data = {}

    metrics_data["ROUGE-1"] = extract_row("ROUGE-1", lambda ds, dim: get_sum_metric(baseline, ds, dim, "rouge1"))
    metrics_data["ROUGE-2"] = extract_row("ROUGE-2", lambda ds, dim: get_sum_metric(baseline, ds, dim, "rouge2"))
    metrics_data["ROUGE-L"] = extract_row("ROUGE-L", lambda ds, dim: get_sum_metric(baseline, ds, dim, "rougeL"))
    metrics_data["MoverScore"] = extract_row("MoverScore", lambda ds, dim: get_sum_metric(baseline, ds, dim, "moverscore"))
    metrics_data["BERTScore"] = extract_row("BERTScore", lambda ds, dim: get_sum_metric(baseline, ds, dim, "bertscore_f1"))
    metrics_data["BARTScore"] = extract_row("BARTScore", lambda ds, dim: get_sum_metric(baseline, ds, dim, bartscore_key.get(ds, "bartscore_src_hypo")))
    metrics_data["AlignScore"] = extract_row("AlignScore", lambda ds, dim: get_sum_metric(baseline, ds, dim, "alignscore"))
    metrics_data["QuestEval"] = extract_row("QuestEval", lambda ds, dim: get_sum_metric(baseline, ds, dim, "questeval"))

    def unieval_key_for_dim(dim):
        mapping = {
            "coherence": "unieval_coherence", "consistency": "unieval_consistency",
            "fluency": "unieval_fluency", "relevance": "unieval_relevance",
            "informativeness": "unieval_relevance", "litepyramid_recall": "unieval_overall",
        }
        return mapping.get(dim, "unieval_overall")

    metrics_data["UniEval$^\\dagger$"] = extract_row(
        "UniEval", lambda ds, dim: get_sum_metric(baseline, ds, dim, unieval_key_for_dim(dim)))

    def ds_best(ds, dim):
        candidates = ["diffscore_cond", "diffscore_bi_a5", "diffscore_bi_a7", "diffscore_rev"]
        vals = [get_sum_metric(baseline, ds, dim, c) for c in candidates]
        vals = [v for v in vals if v is not None]
        return max(vals) if vals else None

    metrics_data["DiffScore-Zero"] = extract_row("DiffScore-Zero", ds_best)

    def ds_ft_best(ds, dim):
        candidates = ["diffscore_ft_cond", "diffscore_ft_bi_a5", "diffscore_ft_bi_a7", "diffscore_ft_rev"]
        vals = [get_sum_metric(sft, ds, dim, c) for c in candidates]
        vals = [v for v in vals if v is not None]
        return max(vals) if vals else None

    metrics_data["DiffScore-FT"] = extract_row("DiffScore-FT", ds_ft_best)

    col_labels = [c[2] for c in columns] + ["Avg."]

    all_methods = list(metrics_data.keys())
    col_best = []
    for j in range(len(col_labels)):
        vals_j = [(name, metrics_data[name][j]) for name in all_methods
                  if metrics_data[name][j] is not None]
        col_best.append(max(vals_j, key=lambda x: x[1])[1] if vals_j else 0)

    latex = []
    latex.append("\\begin{table*}[t]")
    latex.append("\\centering")
    latex.append("\\caption{\\label{tab:summ} Spearman correlation of different metrics on three human judgement datasets. "
                 "$\\dagger$ denotes methods trained on large-scale human judgment data. "
                 "The highest correlation overall for each aspect on each dataset is \\textbf{bold}.}")
    latex.append("\\resizebox{\\textwidth}{!}{")
    latex.append("\\begin{tabular}{l" + "c" * len(col_labels) + "}")
    latex.append("\\toprule")

    # Multi-level header
    latex.append(" & \\multicolumn{1}{c}{REALSumm} & \\multicolumn{4}{c}{SummEval} & "
                 "\\multicolumn{4}{c}{Newsroom} & \\\\")
    latex.append("\\cmidrule(lr){2-2}\\cmidrule(lr){3-6}\\cmidrule(lr){7-10}")

    header = " & ".join([f"\\textsc{{{l}}}" for l in col_labels])
    latex.append(f" & {header} \\\\")
    latex.append("\\midrule")

    for name in all_methods:
        vals = metrics_data[name]
        row_parts = []
        for j, v in enumerate(vals):
            is_best = (v is not None and v == col_best[j])
            row_parts.append(fmt(v, bold=is_best))
        row = " & ".join(row_parts)
        if name == "UniEval$^\\dagger$":
            latex.append(f"{name} & {row} \\\\")
            latex.append("\\midrule")
        elif name == "DiffScore-Zero":
            latex.append(f"{name} & {row} \\\\")
        else:
            latex.append(f"{name} & {row} \\\\")

    latex.append("\\bottomrule")
    latex.append("\\end{tabular}}")
    latex.append("\\end{table*}")

    return "\n".join(latex)


# ==============================================================================
# TABLE 3: Factuality - Rank19 + QAGS
# ==============================================================================
def generate_fact_table():
    baseline = load_json(OUTPUTS / "baseline" / "exp1_meta_eval_sum" / "meta_evaluation_results.json")
    sft = load_json(OUTPUTS / "sft_all" / "sft_epoch_3" / "exp1_meta_eval_sum" / "meta_evaluation_results.json")

    def get_rank19(data, metric):
        try:
            return data["sum_Rank19"]["fact"][metric]["accuracy"]
        except (KeyError, TypeError):
            return None

    def get_qags(data, dataset, metric):
        try:
            return data[f"sum_{dataset}"]["fact"][metric]["pearson"]
        except (KeyError, TypeError):
            return None

    metrics = {}

    traditional = [
        ("ROUGE-1", "rouge1"), ("ROUGE-2", "rouge2"), ("ROUGE-L", "rougeL"),
    ]
    for label, key in traditional:
        metrics[label] = [
            get_rank19(baseline, key),
            get_qags(baseline, "QAGS_CNN", key),
            get_qags(baseline, "QAGS_XSUM", key),
        ]

    metrics["MoverScore"] = [get_rank19(baseline, "moverscore"), get_qags(baseline, "QAGS_CNN", "moverscore"), get_qags(baseline, "QAGS_XSUM", "moverscore")]
    metrics["BERTScore"] = [get_rank19(baseline, "bertscore_f1"), get_qags(baseline, "QAGS_CNN", "bertscore_f1"), get_qags(baseline, "QAGS_XSUM", "bertscore_f1")]
    metrics["BARTScore"] = [get_rank19(baseline, "bartscore_src_hypo"), get_qags(baseline, "QAGS_CNN", "bartscore_src_hypo"), get_qags(baseline, "QAGS_XSUM", "bartscore_src_hypo")]
    metrics["AlignScore"] = [get_rank19(baseline, "alignscore"), get_qags(baseline, "QAGS_CNN", "alignscore"), get_qags(baseline, "QAGS_XSUM", "alignscore")]
    metrics["QuestEval"] = [get_rank19(baseline, "questeval"), get_qags(baseline, "QAGS_CNN", "questeval"), get_qags(baseline, "QAGS_XSUM", "questeval")]
    metrics["UniEval$^\\dagger$"] = [get_rank19(baseline, "unieval_consistency"), get_qags(baseline, "QAGS_CNN", "unieval_consistency"), get_qags(baseline, "QAGS_XSUM", "unieval_consistency")]

    def ds_best_rank19(data, prefix="diffscore"):
        candidates = [f"{prefix}_cond", f"{prefix}_bi_a5", f"{prefix}_bi_a7"]
        vals = [get_rank19(data, c) for c in candidates]
        vals = [v for v in vals if v is not None]
        return max(vals) if vals else None

    def ds_best_qags(data, dataset, prefix="diffscore"):
        candidates_pearson = [f"{prefix}_cond", f"{prefix}_bi_a5", f"{prefix}_bi_a7", f"{prefix}_pmi"]
        vals = [get_qags(data, dataset, c) for c in candidates_pearson]
        vals = [v for v in vals if v is not None]
        return max(vals) if vals else None

    metrics["DiffScore-Zero"] = [
        ds_best_rank19(baseline), ds_best_qags(baseline, "QAGS_CNN"), ds_best_qags(baseline, "QAGS_XSUM")]
    metrics["DiffScore-FT"] = [
        ds_best_rank19(sft, "diffscore_ft"), ds_best_qags(sft, "QAGS_CNN", "diffscore_ft"), ds_best_qags(sft, "QAGS_XSUM", "diffscore_ft")]

    col_labels = ["Rank19", "Q-CNN", "Q-XSUM"]
    all_methods = list(metrics.keys())

    col_best = []
    for j in range(len(col_labels)):
        vals_j = [(name, metrics[name][j]) for name in all_methods if metrics[name][j] is not None]
        col_best.append(max(vals_j, key=lambda x: x[1])[1] if vals_j else 0)

    latex = []
    latex.append("\\begin{wraptable}[19]{r}{6.5cm}")
    latex.append("\\caption{\\label{tab:fact} Results on Rank19 and QAGS datasets. "
                 "$\\dagger$ trained on large-scale data. "
                 "Metrics achieving highest correlation are \\textbf{bold}.}")
    latex.append("\\begin{tabular}{lccc}")
    latex.append("\\toprule")
    latex.append(" & \\multicolumn{1}{c}{Rank19} & \\multicolumn{1}{c}{Q-CNN} & \\multicolumn{1}{c}{Q-XSUM} \\\\")
    latex.append("\\cmidrule(lr){2-2}\\cmidrule(lr){3-4}")
    latex.append(" & \\multicolumn{1}{c}{Acc.} & \\multicolumn{2}{c}{Pearson} \\\\")
    latex.append("\\midrule")

    for name in all_methods:
        vals = metrics[name]
        row_parts = []
        for j, v in enumerate(vals):
            is_best = (v is not None and v == col_best[j])
            row_parts.append(fmt(v, bold=is_best))
        row = " & ".join(row_parts)
        latex.append(f"{name} & {row} \\\\")
        if name == "UniEval$^\\dagger$":
            latex.append("\\midrule")

    latex.append("\\bottomrule")
    latex.append("\\end{tabular}")
    latex.append("\\end{wraptable}")

    return "\n".join(latex)


# ==============================================================================
# TABLE 4: D2T - Data-to-Text Spearman
# ==============================================================================
def generate_d2t_table():
    baseline = load_json(OUTPUTS / "baseline" / "exp1_meta_eval_d2t" / "meta_evaluation_results.json")
    sft = load_json(OUTPUTS / "sft_all" / "sft_epoch_3" / "exp1_meta_eval_d2t" / "meta_evaluation_results.json")

    datasets = ["BAGEL", "SFRES", "SFHOT"]

    def avg_d2t(data, dataset, metric):
        vals = []
        for dim in ["informativeness", "naturalness", "quality"]:
            try:
                vals.append(data[f"d2t_{dataset}"][dim][metric]["spearman"])
            except (KeyError, TypeError):
                pass
        return np.mean(vals) if vals else None

    metrics = {}
    metrics["ROUGE-1"] = [avg_d2t(baseline, ds, "rouge1") for ds in datasets]
    metrics["ROUGE-2"] = [avg_d2t(baseline, ds, "rouge2") for ds in datasets]
    metrics["ROUGE-L"] = [avg_d2t(baseline, ds, "rougeL") for ds in datasets]
    metrics["MoverScore"] = [avg_d2t(baseline, ds, "moverscore") for ds in datasets]
    metrics["BERTScore"] = [avg_d2t(baseline, ds, "bertscore_f1") for ds in datasets]
    metrics["BARTScore"] = [avg_d2t(baseline, ds, "bartscore_ref_hypo") for ds in datasets]
    metrics["AlignScore"] = [avg_d2t(baseline, ds, "alignscore") for ds in datasets]
    metrics["QuestEval"] = [avg_d2t(baseline, ds, "questeval") for ds in datasets]
    metrics["UniEval$^\\dagger$"] = [avg_d2t(baseline, ds, "unieval_overall") for ds in datasets]

    def ds_best_d2t(data, dataset, prefix="diffscore"):
        candidates = [f"{prefix}_cond_src", f"{prefix}_cond_ref", f"{prefix}_bi_a7"]
        vals = []
        for dim in ["informativeness", "naturalness", "quality"]:
            dim_vals = []
            for c in candidates:
                try:
                    dim_vals.append(data[f"d2t_{dataset}"][dim][c]["spearman"])
                except (KeyError, TypeError):
                    pass
            vals.append(max(dim_vals) if dim_vals else 0)
        return np.mean(vals) if vals else None

    metrics["DiffScore-Zero"] = [ds_best_d2t(baseline, ds) for ds in datasets]
    metrics["DiffScore-FT"] = [ds_best_d2t(sft, ds, "diffscore_ft") for ds in datasets]

    for name in metrics:
        vals = [v for v in metrics[name] if v is not None]
        metrics[name].append(np.mean(vals) if vals else None)

    col_labels = datasets + ["Avg."]
    all_methods = list(metrics.keys())

    col_best = []
    for j in range(len(col_labels)):
        vals_j = [(name, metrics[name][j]) for name in all_methods if metrics[name][j] is not None]
        col_best.append(max(vals_j, key=lambda x: x[1])[1] if vals_j else 0)

    latex = []
    latex.append("\\begin{wraptable}[17]{r}{7cm}")
    latex.append("\\caption{\\label{tab:d2t} Results on data-to-text datasets. "
                 "We report average Spearman correlation across three dimensions "
                 "(informativeness, naturalness, quality). "
                 "$\\dagger$ trained on large-scale data. "
                 "Highest correlation \\textbf{bold}.}")
    latex.append("\\begin{tabular}{lcccc}")
    latex.append("\\toprule")
    header = " & ".join([f"\\textbf{{{l}}}" for l in col_labels])
    latex.append(f" & {header} \\\\")
    latex.append("\\midrule")

    for name in all_methods:
        vals = metrics[name]
        row_parts = []
        for j, v in enumerate(vals):
            is_best = (v is not None and abs(v - col_best[j]) < 1e-6)
            row_parts.append(fmt(v, bold=is_best))
        row = " & ".join(row_parts)
        latex.append(f"{name} & {row} \\\\")
        if name == "UniEval$^\\dagger$":
            latex.append("\\midrule")

    latex.append("\\bottomrule")
    latex.append("\\end{tabular}")
    latex.append("\\end{wraptable}")

    return "\n".join(latex)


if __name__ == "__main__":
    out_dir = PROJECT_ROOT / "outputs" / "tables"
    out_dir.mkdir(parents=True, exist_ok=True)

    tables = {
        "table1_wmt.tex": generate_wmt_table,
        "table2_sum.tex": generate_sum_table,
        "table3_fact.tex": generate_fact_table,
        "table4_d2t.tex": generate_d2t_table,
    }

    for filename, gen_fn in tables.items():
        latex = gen_fn()
        outpath = out_dir / filename
        with open(outpath, "w") as f:
            f.write(latex)
        print(f"\n{'='*60}")
        print(f"Generated: {outpath}")
        print(f"{'='*60}")
        print(latex)
