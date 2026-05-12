"""
WebNLG 2020 dataset loader.

Data-to-text evaluation with 5 dimensions:
Correctness, Data Coverage, Fluency, Relevance, Text Structure.
"""

import json
import os
import logging
from typing import List, Dict, Optional
from dataclasses import dataclass, field
import numpy as np
import random

logger = logging.getLogger(__name__)


WEBNLG_DIMENSIONS = [
    "correctness", "data_coverage", "fluency", "relevance", "text_structure"
]

PROMPT_TEMPLATES = {
    "default": {
        "prefix": "Data: ",
        "mid": "\n\nText: ",
        "suffix": "",
    },
    "v2": {
        "prefix": "Given the following structured data:\n",
        "mid": "\n\nGenerate a text description:\n",
        "suffix": "",
    },
}


@dataclass
class WebNLGSample:
    source: str           # triples / structured data
    candidate: str        # generated text
    reference: str
    system_id: str
    sample_id: str
    scores: Dict[str, float] = field(default_factory=dict)


class WebNLGDataset:
    """Loader for WebNLG 2020 benchmark."""

    def __init__(self, data_path: Optional[str] = None):
        self.samples: List[WebNLGSample] = []
        if data_path:
            self.load_from_file(data_path)

    def load_from_file(self, path: str):
        self.samples = []
        with open(path, "r") as f:
            if path.endswith(".jsonl"):
                for line in f:
                    self._parse_entry(json.loads(line))
            else:
                data = json.load(f)
                entries = data if isinstance(data, list) else data.get("data", [])
                for entry in entries:
                    self._parse_entry(entry)
        logger.info(f"Loaded {len(self.samples)} WebNLG samples")

    def load_from_huggingface(self):
        """Load WebNLG data from HuggingFace."""
        from datasets import load_dataset
        ds = load_dataset("teven/webnlg_2020_human_eval", split="train", trust_remote_code=True)

        self.samples = []
        for i, row in enumerate(ds):
            rdf_content = row.get("rdf", "")
            source = self._process_rdf(rdf_content)

            candidate_text = row.get("prediction", row.get("text", ""))
            references = row.get("references", row.get("target", [""]))
            if isinstance(references, list):
                reference = references
            else:
                reference = [references]

            scores = {}
            for dim in WEBNLG_DIMENSIONS:
                val = row.get(dim, 0.0)
                if val is None:
                    scores[dim] = 0.0
                elif isinstance(val, (list, np.ndarray)):
                    scores[dim] = float(np.mean(val))
                else:
                    scores[dim] = float(val)

            self.samples.append(WebNLGSample(
                source=source,
                candidate=candidate_text,
                reference=reference,
                system_id=row.get("team", "unknown"),
                sample_id=str(i),
                scores=scores,
            ))
        logger.info(f"Loaded {len(self.samples)} WebNLG samples from HuggingFace")

    @staticmethod
    def _process_rdf(rdf_content) -> str:
        """Convert RDF triples into a flat string representation."""
        if isinstance(rdf_content, list):
            parts = []
            for item in rdf_content:
                if isinstance(item, dict):
                    s = item.get("subject", "").strip()
                    p = item.get("property", "").strip()
                    o = item.get("object", "").strip()
                    if s and p and o:
                        parts.append(f"{s} | {p} | {o}")
                elif isinstance(item, str):
                    parts.append(item.strip())
            return " , ".join(parts)
        return str(rdf_content)

    def _parse_entry(self, entry: dict):
        scores = {}
        for dim in WEBNLG_DIMENSIONS:
            val = entry[dim]
            scores[dim] = float(np.mean(val)) if isinstance(val, list) else float(val)

        source = entry.get("rdf", entry.get("source", entry.get("triples", "")))
        if isinstance(source, list):
            processed_triples = []
            for triple_dict in source:
                # 提取subject、property、object（使用get避免键不存在报错）
                s_part = triple_dict.get("subject", "").strip()
                p_part = triple_dict.get("property", "").strip()
                o_part = triple_dict.get("object", "").strip()
                
                # 拼接成 (subject, property, object) 格式（仅当三个字段都非空时）
                if s_part and p_part and o_part:
                    processed_triples.append(f"{s_part} | {p_part} | {o_part}")
            
            # 3. 用 | 连接所有处理后的三元组
            source = " , ".join(processed_triples)

        references = entry.get("references", [])

        self.samples.append(WebNLGSample(
            source=str(source),
            candidate=entry.get("candidates", entry.get("candidate", entry.get("prediction", ""))),
            reference=references[0],
            system_id=entry.get("team", "unknown"),
            sample_id=entry.get("ID", entry.get("sample_id", "unknown")),
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
        return {dim: self.get_human_scores(dim) for dim in WEBNLG_DIMENSIONS}

    def get_prompt_template(self, variant: str = "default") -> Dict[str, str]:
        return PROMPT_TEMPLATES[variant]

    def __len__(self) -> int:
        return len(self.samples)
