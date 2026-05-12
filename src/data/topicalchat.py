"""
TopicalChat dataset loader.

Dialogue evaluation with 5 dimensions:
Engaging, Natural, Coherent, Uses Knowledge, Overall.
"""

import json
import os
import logging
from typing import List, Dict, Optional
from dataclasses import dataclass, field
import numpy as np

logger = logging.getLogger(__name__)

TOPICALCHAT_DIMENSIONS = [
    "Understandable", "Natural", "Maintains Context", "Engaging", "Uses Knowledge", "Overall"
]

PROMPT_TEMPLATES = {
    "default": {
        "prefix": "Below is a dialogue context and a response. Read both carefully.\n\nDialogue context:\n",
        "mid": "\n\nResponse to evaluate:\n",
        "suffix": "",
    },
    "v2": {
        "prefix": "The following is a conversation:\n",
        "mid": "\n\nThe last response is:\n",
        "suffix": "",
    },
    "v3": {
        "prefix": "Recent dialogue:\n",
        "mid": "\n\nResponse:\n",
        "suffix": "",
    },
    "simple": {
        "prefix": "Context: ",
        "mid": "\n\nResponse: ",
        "suffix": "",
    },
}


@dataclass
class TopicalChatSample:
    context: str          # dialogue history
    fact: str             # fact
    candidate: str        # system response
    reference: str
    system_id: str
    dialogue_id: str
    scores: Dict[str, float] = field(default_factory=dict)


class TopicalChatDataset:
    """Loader for TopicalChat evaluation benchmark."""

    def __init__(self, data_path: Optional[str] = None):
        self.samples: List[TopicalChatSample] = []
        if data_path:
            ### data downloaded from "https://shikib.com/tc_usr_data.json" and data intro from "https://shikib.com/usr".
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
        logger.info(f"Loaded {len(self.samples)} TopicalChat samples")

    def load_from_huggingface(self):
        """Load TopicalChat USR data from HuggingFace."""
        from datasets import load_dataset
        ## No longer valid for the dataset in huggingface.
        ds = load_dataset(
            "McGill-NLP/topical-chat-usr",
            split="test",
            trust_remote_code=True,
        )
        self.samples = []
        for i, row in enumerate(ds):
            context = row.get("context", "")
            if isinstance(context, list):
                context = "\n".join(context)

            scores = {}
            for dim in TOPICALCHAT_DIMENSIONS:
                for key in [dim, dim.replace("_", " "), f"ann_{dim}"]:
                    if key in row:
                        val = row[key]
                        scores[dim] = float(np.mean(val)) if isinstance(val, list) else float(val)
                        break

            self.samples.append(TopicalChatSample(
                context=context,
                fact=row.get("fact", None),
                candidate=row.get("response", row.get("candidate", "")),
                reference=row.get("reference", ""),
                system_id=row.get("system", "unknown"),
                dialogue_id=str(row.get("dialogue_id", i)),
                scores=scores,
            ))
        logger.info(f"Loaded {len(self.samples)} TopicalChat samples from HuggingFace")

    def _parse_entry(self, entry: dict):
        # Get context (either "context" or "history")
        context = entry.get("context", entry.get("history", ""))
        if isinstance(context, list):
            context = "\n".join(context)

        # Get fact (if available)
        fact = entry.get("fact", None)

        # Get responses (default to "response" if not available)
        responses = entry.get("responses", [])

        # Find the "Original Ground Truth" reference response
        reference = next((response["response"] for response in responses if response["model"] == "Original Ground Truth"), None)

        # Loop through other responses and process them
        for response in responses:
            if response["model"] == "Original Ground Truth":
                continue  # Skip the ground truth response

            # Extract candidate response
            candidate = response['response']
            
            scores = {}
            for dim in TOPICALCHAT_DIMENSIONS:
                val = response.get(dim)
                if val is None:
                    scores[dim] = 0.0
                elif isinstance(val, list):
                    scores[dim] = float(np.mean(val))
                else:
                    scores[dim] = float(val)

            # Append processed sample to self.samples
            self.samples.append(TopicalChatSample(
                context=context,
                fact=fact,
                candidate=candidate,
                reference=reference,
                system_id=entry.get("system_id", "unknown"),
                dialogue_id=entry.get("dialogue_id", entry.get("id", "unknown")),
                scores=scores,
            ))

    def get_fact(self) -> List[str]:
        return [s.fact for s in self.samples]

    def get_sources(self, max_turns: Optional[int] = None) -> List[str]:
        """Get source contexts, optionally truncated to last N turns.

        Args:
            max_turns: if set, keep only the last N turns of dialogue history
                       to prevent exceeding model context windows.
        """
        if max_turns is None:
            return [s.context for s in self.samples]
        return [self._truncate_context(s.context, max_turns) for s in self.samples]

    @staticmethod
    def _truncate_context(context: str, max_turns: int) -> str:
        """Keep only the last `max_turns` turns from a dialogue context."""
        turns = context.split("\n")
        turns = [t for t in turns if t.strip()]
        if len(turns) <= max_turns:
            return context
        return "\n".join(turns[-max_turns:])

    def get_candidates(self) -> List[str]:
        return [s.candidate for s in self.samples]

    def get_references(self) -> List[str]:
        return [s.reference for s in self.samples]

    def get_human_scores(self, dimension: str) -> np.ndarray:
        return np.array([s.scores.get(dimension, 0.0) for s in self.samples])

    def get_all_human_scores(self) -> Dict[str, np.ndarray]:
        return {dim: self.get_human_scores(dim) for dim in TOPICALCHAT_DIMENSIONS}

    def get_prompt_template(self, variant: str = "default") -> Dict[str, str]:
        return PROMPT_TEMPLATES[variant]

    def __len__(self) -> int:
        return len(self.samples)
