from .bartscore_benchmark import (
    WMTBenchmark,
    SUMBenchmark,
    D2TBenchmark,
    BENCHMARK_REGISTRY,
    load_benchmark,
    get_benchmarks_by_task,
    get_all_benchmark_names,
)

# Legacy loaders (used by experiments 0, 2-8)
from .summeval import SummEvalDataset
from .wmt import WMTDataset
from .webnlg import WebNLGDataset
from .topicalchat import TopicalChatDataset

__all__ = [
    # BartScore-aligned benchmarks (primary)
    "WMTBenchmark", "SUMBenchmark", "D2TBenchmark",
    "BENCHMARK_REGISTRY", "load_benchmark",
    "get_benchmarks_by_task", "get_all_benchmark_names",
    # Legacy
    "SummEvalDataset", "WMTDataset", "WebNLGDataset", "TopicalChatDataset",
]
