"""
Unified data loaders for BartScore-aligned benchmarks.

Three task categories with consistent pickle-based data format:
  - WMT: Machine Translation (preference-based, 7 language pairs)
  - SUM: Text Summarization (6 datasets with varying human metrics)
  - D2T: Data-to-Text Generation (3 datasets)

Data format follows the BartScore paper (Yuan et al., 2021) exactly.
All data is loaded from pre-built data.pkl files in datasets/.
"""

import pickle
import logging
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)


def _read_pickle(path: str) -> dict:
    with open(path, "rb") as f:
        return pickle.load(f)


# ---------------------------------------------------------------------------
# WMT: Machine Translation (preference-based evaluation)
# ---------------------------------------------------------------------------

WMT_LANG_PAIRS = ["de-en", "fi-en", "gu-en", "kk-en", "lt-en", "ru-en", "zh-en"]


class WMTBenchmark:
    """WMT-19 DARR data (preference pairs).

    Structure per doc_id:
        {src, ref, better: {sys_name, sys, scores}, worse: {sys_name, sys, scores}}

    Human annotation is binary preference (better vs worse), not numeric.
    Evaluation: concordance-based Kendall τ.
    """

    def __init__(self, data_path: str, lang_pair: str = ""):
        self.data = _read_pickle(data_path)
        self.lang_pair = lang_pair
        self.doc_ids = list(self.data.keys())
        logger.info(
            f"WMT [{lang_pair}] loaded: {len(self.doc_ids)} preference pairs"
        )

    def get_refs(self) -> List[str]:
        return [self.data[d]["ref"] for d in self.doc_ids]

    def get_srcs(self) -> List[str]:
        return [self.data[d]["src"] for d in self.doc_ids]

    def get_betters(self) -> List[str]:
        return [self.data[d]["better"]["sys"] for d in self.doc_ids]

    def get_worses(self) -> List[str]:
        return [self.data[d]["worse"]["sys"] for d in self.doc_ids]

    def __len__(self) -> int:
        return len(self.doc_ids)


# ---------------------------------------------------------------------------
# SUM: Text Summarization
# ---------------------------------------------------------------------------

SUM_BENCHMARKS = {
    "SummEval": {
        "human_metrics": ["coherence", "consistency", "fluency", "relevance"],
        "multi_ref": True,
        "eval_type": "document_correlation",
    },
    "Newsroom": {
        "human_metrics": ["coherence", "fluency", "informativeness", "relevance"],
        "multi_ref": False,
        "eval_type": "document_correlation",
    },
    "REALSumm": {
        "human_metrics": ["litepyramid_recall"],
        "multi_ref": False,
        "eval_type": "document_correlation",
    },
    "Rank19": {
        "human_metrics": ["fact"],
        "multi_ref": False,
        "eval_type": "accuracy",
    },
    "QAGS_CNN": {
        "human_metrics": ["fact"],
        "multi_ref": False,
        "eval_type": "fact_pearson",
    },
    "QAGS_XSUM": {
        "human_metrics": ["fact"],
        "multi_ref": False,
        "eval_type": "fact_pearson",
    },
}


class SUMBenchmark:
    """Summarization benchmark data.

    Structure per doc_id:
        {src, ref_summ (or ref_summs), sys_summs: {sys_name: {sys_summ, scores: {...}}}}

    Different sub-datasets use different human metrics and evaluation:
      - SummEval/Newsroom/REALSumm: per-document Spearman/Kendall across systems
      - Rank19: accuracy (correct vs incorrect)
      - QAGS: Pearson with factuality
    """

    def __init__(self, data_path: str, benchmark_name: str):
        self.data = _read_pickle(data_path)
        self.benchmark_name = benchmark_name
        self.doc_ids = list(self.data.keys())

        cfg = SUM_BENCHMARKS.get(benchmark_name, {})
        self.human_metrics = cfg.get("human_metrics", [])
        self.multi_ref = cfg.get("multi_ref", False)
        self.eval_type = cfg.get("eval_type", "document_correlation")

        self.sys_names = self._get_sys_names()
        logger.info(
            f"SUM [{benchmark_name}] loaded: {len(self.doc_ids)} docs, "
            f"{len(self.sys_names)} systems, eval_type={self.eval_type}"
        )

    def _get_sys_names(self) -> List[Any]:
        first_id = self.doc_ids[0]
        return list(self.data[first_id]["sys_summs"].keys())

    def get_src_lines(self) -> List[str]:
        return [self.data[d]["src"] for d in self.doc_ids]

    def get_single_ref_lines(self) -> List[str]:
        """Single reference per document."""
        refs = []
        for d in self.doc_ids:
            if "ref_summ" in self.data[d]:
                refs.append(self.data[d]["ref_summ"])
            elif "ref_summs" in self.data[d]:
                refs.append(self.data[d]["ref_summs"][0])
            else:
                refs.append("")
        return refs

    def get_multi_ref_lines(self) -> List[List[str]]:
        """Multiple references per document (returns list of lists)."""
        refs = []
        for d in self.doc_ids:
            if "ref_summs" in self.data[d]:
                refs.append(self.data[d]["ref_summs"])
            elif "ref_summ" in self.data[d]:
                refs.append([self.data[d]["ref_summ"]])
            else:
                refs.append([""])
        return refs

    def get_ref_count(self) -> int:
        if self.multi_ref:
            refs = self.get_multi_ref_lines()
            return len(refs[0]) if refs else 1
        return 1

    def get_sys_lines(self, sys_name) -> List[str]:
        return [self.data[d]["sys_summs"][sys_name]["sys_summ"] for d in self.doc_ids]

    def get_human_scores_for_sys(self, sys_name, human_metric: str) -> List[float]:
        return [
            self.data[d]["sys_summs"][sys_name]["scores"].get(human_metric, 0.0)
            for d in self.doc_ids
        ]

    def iter_all_samples(self):
        """Iterate over all (doc_id, sys_name, src, sys_summ) tuples."""
        for d in self.doc_ids:
            src = self.data[d]["src"]
            for sys_name in self.data[d]["sys_summs"]:
                sys_summ = self.data[d]["sys_summs"][sys_name]["sys_summ"]
                yield d, sys_name, src, sys_summ

    def store_metric_score(self, doc_id, sys_name, metric_name: str, score: float):
        """Store a computed metric score back into the data dict."""
        self.data[doc_id]["sys_summs"][sys_name]["scores"][metric_name] = score

    def __len__(self) -> int:
        return len(self.doc_ids)


# ---------------------------------------------------------------------------
# D2T: Data-to-Text Generation
# ---------------------------------------------------------------------------

D2T_BENCHMARKS = {
    "BAGEL": {"human_metrics": ["informativeness", "naturalness", "quality"]},
    "SFHOT": {"human_metrics": ["informativeness", "naturalness", "quality"]},
    "SFRES": {"human_metrics": ["informativeness", "naturalness", "quality"]},
}


class D2TBenchmark:
    """Data-to-Text benchmark data.

    Structure per doc_id:
        {src, sys_summ, ref_summs: [...], scores: {informativeness, naturalness, quality}}

    Evaluation: document-level Spearman/Kendall.
    """

    def __init__(self, data_path: str, benchmark_name: str):
        self.data = _read_pickle(data_path)
        self.benchmark_name = benchmark_name
        self.doc_ids = list(self.data.keys())

        cfg = D2T_BENCHMARKS.get(benchmark_name, {})
        self.human_metrics = cfg.get(
            "human_metrics", ["informativeness", "naturalness", "quality"]
        )
        logger.info(
            f"D2T [{benchmark_name}] loaded: {len(self.doc_ids)} documents"
        )

    def get_src_lines(self) -> List[str]:
        return [str(self.data[d]["src"]) for d in self.doc_ids]

    def get_sys_summs(self) -> List[str]:
        return [self.data[d]["sys_summ"] for d in self.doc_ids]

    def get_ref_summs(self) -> List[List[str]]:
        return [self.data[d]["ref_summs"] for d in self.doc_ids]

    def get_human_scores(self, human_metric: str) -> List[float]:
        return [
            self.data[d]["scores"].get(human_metric, 0.0) for d in self.doc_ids
        ]

    def store_metric_score(self, doc_id, metric_name: str, score: float):
        """Store a computed metric score back into the data dict."""
        self.data[doc_id]["scores"][metric_name] = score

    def __len__(self) -> int:
        return len(self.doc_ids)


# ---------------------------------------------------------------------------
# Registry: all benchmarks with paths and metadata
# ---------------------------------------------------------------------------

BENCHMARK_REGISTRY = {}

# WMT benchmarks
for _lp in WMT_LANG_PAIRS:
    BENCHMARK_REGISTRY[f"wmt_{_lp}"] = {
        "task": "WMT",
        "class": WMTBenchmark,
        "data_path": f"datasets/WMT/{_lp}/data.pkl",
        "lang_pair": _lp,
        "eval_type": "preference_kendall",
        "prompt_template": {
            "prefix": "Reference: ",
            "mid": "\n\nTranslation: ",
            "suffix": "",
            "system": "Below is a reference translation and a candidate translation.",
        },
    }

# SUM benchmarks
for _name, _cfg in SUM_BENCHMARKS.items():
    BENCHMARK_REGISTRY[f"sum_{_name}"] = {
        "task": "SUM",
        "class": SUMBenchmark,
        "data_path": f"datasets/SUM/{_name}/data.pkl",
        "benchmark_name": _name,
        "human_metrics": _cfg["human_metrics"],
        "eval_type": _cfg["eval_type"],
        "prompt_template": {
            "prefix": "Document: ",
            "mid": "\n\nSummary: ",
            "suffix": "",
            "system": "Below is a source document. The following is a summary of the document.",
        },
    }

# D2T benchmarks
for _name, _cfg in D2T_BENCHMARKS.items():
    BENCHMARK_REGISTRY[f"d2t_{_name}"] = {
        "task": "D2T",
        "class": D2TBenchmark,
        "data_path": f"datasets/D2T/{_name}/data.pkl",
        "benchmark_name": _name,
        "human_metrics": _cfg["human_metrics"],
        "eval_type": "document_correlation",
        "prompt_template": {
            "prefix": "Data: ",
            "mid": "\n\nText: ",
            "suffix": "",
            "system": "Below is structured data. The following is a natural language description of the data.",
        },
    }


def load_benchmark(name: str, base_dir: str = ".") -> Any:
    """Load a benchmark by registry name.

    Args:
        name: key from BENCHMARK_REGISTRY (e.g. 'wmt_de-en', 'sum_SummEval', 'd2t_BAGEL')
        base_dir: project root directory

    Returns:
        WMTBenchmark, SUMBenchmark, or D2TBenchmark instance
    """
    import os

    if name not in BENCHMARK_REGISTRY:
        raise ValueError(
            f"Unknown benchmark '{name}'. Available: {list(BENCHMARK_REGISTRY.keys())}"
        )

    cfg = BENCHMARK_REGISTRY[name]
    data_path = os.path.join(base_dir, cfg["data_path"])

    if cfg["task"] == "WMT":
        return cfg["class"](data_path, lang_pair=cfg["lang_pair"])
    elif cfg["task"] == "SUM":
        return cfg["class"](data_path, benchmark_name=cfg["benchmark_name"])
    elif cfg["task"] == "D2T":
        return cfg["class"](data_path, benchmark_name=cfg["benchmark_name"])
    else:
        raise ValueError(f"Unknown task type: {cfg['task']}")


def get_benchmarks_by_task(task: str) -> List[str]:
    """Get all benchmark names for a given task ('WMT', 'SUM', 'D2T')."""
    return [k for k, v in BENCHMARK_REGISTRY.items() if v["task"] == task]


def get_all_benchmark_names() -> List[str]:
    return list(BENCHMARK_REGISTRY.keys())
