"""
WMT22 Metrics Shared Task dataset loader.

Supports zh-en and en-de language pairs with MQM annotations.
"""

import json
import os
import logging
from typing import List, Dict, Optional
from dataclasses import dataclass, field
import numpy as np

logger = logging.getLogger(__name__)

PROMPT_TEMPLATES = {
    "zh-en": {
        "prefix": "Source (Chinese): ",
        "mid": "\n\nTranslation (English): ",
        "suffix": "",
    },
    "en-de": {
        "prefix": "Source (English): ",
        "mid": "\n\nTranslation (German): ",
        "suffix": "",
    },
    "generic": {
        "prefix": "Source: ",
        "mid": "\n\nTranslation: ",
        "suffix": "",
    },
}


@dataclass
class WMTSample:
    source: str
    candidate: str
    reference: str
    system_id: str
    segment_id: str
    lang_pair: str
    mqm_score: float = 0.0
    raw_score: Optional[float] = None


class WMTDataset:
    """Loader for WMT Metrics shared task data."""

    def __init__(self, data_path: Optional[str] = None, lang_pair: str = "zh-en"):
        self.lang_pair = lang_pair
        self.samples: List[WMTSample] = []
        if data_path:
            self.load_from_file(data_path)

    def load_from_file(self, path: str):
        self.samples = []
        with open(path, "r") as f:
            if path.endswith(".jsonl"):
                for line in f:
                    entry = json.loads(line)
                    self._parse_entry(entry)
            else:
                data = json.load(f)
                entries = data if isinstance(data, list) else data.get("data", [])
                for entry in entries:
                    self._parse_entry(entry)
        logger.info(f"Loaded {len(self.samples)} WMT samples ({self.lang_pair})")

    def load_from_huggingface(self, year: int = 2022):
        """Load WMT metrics data from HuggingFace."""
        from datasets import load_dataset

        try:
            ds = load_dataset(
                f"google/wmt{year % 100}_metrics_data",
                self.lang_pair,
                split="test",
            )
        except Exception:
            ds = load_dataset(
                "RicardoRei/wmt-mqm-human-evaluation",
                split="train",
            )
            ds = ds.filter(lambda x: x.get("lp", "") == self.lang_pair)

        self.samples = []
        for row in ds:
            self.samples.append(WMTSample(
                source=row.get("src", ""),
                candidate=row.get("mt", row.get("candidate", "")),
                reference=row.get("ref", row.get("reference", "")),
                system_id=row.get("system", "unknown"),
                segment_id=str(row.get("seg_id", row.get("segment_id", ""))),
                lang_pair=self.lang_pair,
                mqm_score=float(row.get("score", row.get("mqm", 0.0))),
                raw_score=row.get("raw_score", None),
            ))
        logger.info(f"Loaded {len(self.samples)} WMT samples from HuggingFace")

    def _parse_entry(self, entry: dict):
        self.samples.append(WMTSample(
            source=entry.get("src", entry.get("source", "")),
            candidate=entry.get("mt", entry.get("candidate", entry.get("translation", ""))),
            reference=entry.get("ref", entry.get("reference", "")),
            system_id=entry.get("system", "unknown"),
            segment_id=str(entry.get("seg_id", entry.get("segment_id", ""))),
            lang_pair=entry.get("lp", self.lang_pair),
            mqm_score=float(entry.get("score", entry.get("mqm", 0.0))),
        ))

    def get_sources(self) -> List[str]:
        return [s.source for s in self.samples]

    def get_candidates(self) -> List[str]:
        return [s.candidate for s in self.samples]

    def get_references(self) -> List[str]:
        return [s.reference for s in self.samples]

    def get_human_scores(self) -> np.ndarray:
        return np.array([s.mqm_score for s in self.samples])

    def get_system_level_scores(self) -> Dict[str, float]:
        from collections import defaultdict
        system_scores = defaultdict(list)
        for s in self.samples:
            system_scores[s.system_id].append(s.mqm_score)
        return {sid: float(np.mean(scores)) for sid, scores in system_scores.items()}

    def get_prompt_template(self) -> Dict[str, str]:
        return PROMPT_TEMPLATES.get(self.lang_pair, PROMPT_TEMPLATES["generic"])

    def __len__(self) -> int:
        return len(self.samples)
