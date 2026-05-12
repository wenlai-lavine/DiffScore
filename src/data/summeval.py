"""
SummEval dataset loader.

SummEval (Fabbri et al., 2021) contains 1600 summaries (16 systems x 100 source docs)
with human annotations on 4 dimensions: Coherence, Consistency, Fluency, Relevance.
"""

import json
import os
import logging
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
import numpy as np
import random

logger = logging.getLogger(__name__)

SUMMEVAL_DIMENSIONS = ["coherence", "consistency", "fluency", "relevance"]

PROMPT_TEMPLATES = {
    "default": {
        "prefix": "Document: ",
        "mid": "\n\nSummary: ",
        "suffix": "",
    },
    "v2": {
        "prefix": "Given the following document:\n",
        "mid": "\n\nThe summary is:\n",
        "suffix": "",
    },
    "v3": {
        "prefix": "Article: ",
        "mid": "\n\nPlease evaluate the following summary: ",
        "suffix": "",
    },
    "v4": {
        "prefix": "Source document: ",
        "mid": "\nGenerated summary: ",
        "suffix": "",
    },
    "v5": {
        "prefix": "",
        "mid": "\nTL;DR: ",
        "suffix": "",
    },
    "instruct": {
        "prefix": "Below is a source document and its summary. Read both carefully.\n\nSource document:\n",
        "mid": "\n\nSummary:\n",
        "suffix": "",
    },
    "instruct_v2": {
        "prefix": "The following is a document with a corresponding summary.\n\nDocument:\n",
        "mid": "\n\nCorresponding summary:\n",
        "suffix": "",
    },
}


@dataclass
class SummEvalSample:
    source: str
    candidate: str
    # reference: str
    reference: List[str]
    system_id: str
    doc_id: str
    scores: Dict[str, float] = field(default_factory=dict)


class SummEvalDataset:
    """Loader for SummEval benchmark."""

    HF_DATASET = "mteb/summeval"

    def __init__(self, data_path: Optional[str] = None):
        self.samples: List[SummEvalSample] = []
        if data_path:
            self.load_from_file(data_path)

    def load_from_file(self, path: str):
        """Load from local JSON/JSONL file."""
        self.samples = []
        with open(path, "r") as f:
            if path.endswith(".jsonl"):
                for line in f:
                    entry = json.loads(line)
                    self._parse_entry(entry)
            else:
                data = json.load(f)
                if isinstance(data, list):
                    for entry in data:
                        self._parse_entry(entry)
                else:
                    for entry in data.get("data", data.get("rows", [])):
                        self._parse_entry(entry)
        logger.info(f"Loaded {len(self.samples)} samples from {path}")

    def load_from_huggingface(self):
        """Load from HuggingFace datasets."""
        # Note: the length between human_smarries and machine_smarries is different
        from datasets import load_dataset

        ds = load_dataset(self.HF_DATASET, split="test")
        self.samples = []
        for i, row in enumerate(ds):
            source = row.get("text", row.get("source", ""))
            references = row.get("human_summaries", []) # type: list
            machine_summaries = row.get("machine_summaries", []) # type: list

            coherence_scores = row.get("coherence", []) # type: list
            consistency_scores = row.get("consistency", []) # type: list
            fluency_scores = row.get("fluency", []) # type: list
            relevance_scores = row.get("relevance", []) # type: list

            for j, summary in enumerate(machine_summaries):
                scores = {}
                if j < len(coherence_scores):
                    scores["coherence"] = float(np.mean(coherence_scores[j])) if isinstance(coherence_scores[j], list) else float(coherence_scores[j])
                if j < len(consistency_scores):
                    scores["consistency"] = float(np.mean(consistency_scores[j])) if isinstance(consistency_scores[j], list) else float(consistency_scores[j])
                if j < len(fluency_scores):
                    scores["fluency"] = float(np.mean(fluency_scores[j])) if isinstance(fluency_scores[j], list) else float(fluency_scores[j])
                if j < len(relevance_scores):
                    scores["relevance"] = float(np.mean(relevance_scores[j])) if isinstance(relevance_scores[j], list) else float(relevance_scores[j])

                self.samples.append(SummEvalSample(
                    source=source,
                    candidate=summary,
                    # reference=references[random.randint(0, len(references)) - 1],
                    reference=references,
                    system_id=f"sys_{j}",
                    doc_id=f"doc_{i}",
                    scores=scores,
                ))

        logger.info(f"Loaded {len(self.samples)} samples from HuggingFace")

    def _parse_entry(self, entry: dict):
        scores = {}
        for dim in SUMMEVAL_DIMENSIONS:
            if dim in entry:
                val = entry[dim]
                scores[dim] = float(np.mean(val)) if isinstance(val, list) else float(val)
            elif f"expert_{dim}" in entry:
                val = entry[f"expert_{dim}"]
                scores[dim] = float(np.mean(val)) if isinstance(val, list) else float(val)

        self.samples.append(SummEvalSample(
            source=entry.get("text", entry.get("source", entry.get("src", ""))),
            candidate=entry.get("decoded", entry.get("candidate", entry.get("summary", ""))),
            reference=entry.get("reference", entry.get("human_reference", "")),
            system_id=entry.get("model_id", entry.get("system_id", "unknown")),
            doc_id=entry.get("id", entry.get("doc_id", "unknown")),
            scores=scores,
        ))

    def get_sources(self) -> List[str]:
        return [s.source for s in self.samples]

    def get_candidates(self) -> List[str]:
        return [s.candidate for s in self.samples]

    def get_references(self) -> List[str]:
        return [s.reference for s in self.samples]

    def get_human_scores(self, dimension: str) -> np.ndarray:
        return np.array([s.scores.get(dimension, 0.0) for s in self.samples])

    def get_all_human_scores(self) -> Dict[str, np.ndarray]:
        return {dim: self.get_human_scores(dim) for dim in SUMMEVAL_DIMENSIONS}

    def get_system_level_scores(self, dimension: str) -> Dict[str, float]:
        """Aggregate to system-level scores."""
        from collections import defaultdict
        system_scores = defaultdict(list)
        for s in self.samples:
            if dimension in s.scores:
                system_scores[s.system_id].append(s.scores[dimension])
        return {sid: float(np.mean(scores)) for sid, scores in system_scores.items()}

    def get_prompt_template(self, variant: str = "default") -> Dict[str, str]:
        return PROMPT_TEMPLATES[variant]

    def __len__(self) -> int:
        return len(self.samples)
