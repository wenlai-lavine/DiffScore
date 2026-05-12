"""
BARTScore baseline (Yuan et al., 2021).

Implements BARTScore with all configurations:
- bartscore(src, cand): conditional probability p(cand|src)
- bartscore(cand): marginal probability p(cand)
- bartscore_pmi(src, cand): PMI variant for fair comparison with DiffScore-PMI
"""

import torch
import torch.nn.functional as F
from transformers import BartForConditionalGeneration, BartTokenizer
from typing import List, Optional, Dict
import numpy as np
from tqdm import tqdm
import logging

logger = logging.getLogger(__name__)


class BARTScorer:
    """BARTScore implementation for baseline comparison."""

    def __init__(
        self,
        model_name: str = "facebook/bart-large-cnn",
        device: str = "cuda",
        max_length: int = 1024,
        batch_size: int = 8,
    ):
        self.device = device
        self.max_length = max_length
        self.batch_size = batch_size

        logger.info(f"Loading BARTScore model: {model_name}")
        self.tokenizer = BartTokenizer.from_pretrained(model_name)
        self.model = BartForConditionalGeneration.from_pretrained(model_name)
        self.model.eval()
        self.model.to(device)

    @torch.no_grad()
    def _score_pairs(
        self, sources: List[str], targets: List[str]
    ) -> np.ndarray:
        """Compute log p(target | source) for each pair."""
        scores = []

        for i in range(0, len(sources), self.batch_size):
            batch_src = sources[i:i + self.batch_size]
            batch_tgt = targets[i:i + self.batch_size]

            src_enc = self.tokenizer(
                batch_src, return_tensors="pt", padding=True,
                truncation=True, max_length=self.max_length,
            ).to(self.device)

            tgt_enc = self.tokenizer(
                batch_tgt, return_tensors="pt", padding=True,
                truncation=True, max_length=self.max_length,
            ).to(self.device)

            outputs = self.model(
                input_ids=src_enc["input_ids"],
                attention_mask=src_enc["attention_mask"],
                labels=tgt_enc["input_ids"],
            )

            logits = outputs.logits  # (B, L, V)
            log_probs = F.log_softmax(logits, dim=-1)

            tgt_ids = tgt_enc["input_ids"]
            tgt_mask = tgt_enc["attention_mask"]

            token_log_probs = log_probs.gather(
                dim=-1, index=tgt_ids.unsqueeze(-1)
            ).squeeze(-1)

            # Mask padding and average
            token_log_probs = token_log_probs * tgt_mask.float()
            lengths = tgt_mask.float().sum(dim=-1).clamp(min=1)
            avg_scores = token_log_probs.sum(dim=-1) / lengths

            scores.extend(avg_scores.cpu().tolist())

        return np.array(scores)

    @torch.no_grad()
    def _score_marginal(self, texts: List[str]) -> np.ndarray:
        """Compute marginal log p(text) using decoder-only forward."""
        scores = []

        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]

            # Use empty source for marginal probability
            empty_sources = [""] * len(batch)
            src_enc = self.tokenizer(
                empty_sources, return_tensors="pt", padding=True,
                truncation=True, max_length=self.max_length,
            ).to(self.device)

            tgt_enc = self.tokenizer(
                batch, return_tensors="pt", padding=True,
                truncation=True, max_length=self.max_length,
            ).to(self.device)

            outputs = self.model(
                input_ids=src_enc["input_ids"],
                attention_mask=src_enc["attention_mask"],
                labels=tgt_enc["input_ids"],
            )

            logits = outputs.logits
            log_probs = F.log_softmax(logits, dim=-1)

            tgt_ids = tgt_enc["input_ids"]
            tgt_mask = tgt_enc["attention_mask"]

            token_log_probs = log_probs.gather(
                dim=-1, index=tgt_ids.unsqueeze(-1)
            ).squeeze(-1)

            token_log_probs = token_log_probs * tgt_mask.float()
            lengths = tgt_mask.float().sum(dim=-1).clamp(min=1)
            avg_scores = token_log_probs.sum(dim=-1) / lengths

            scores.extend(avg_scores.cpu().tolist())

        return np.array(scores)

    def score(
        self,
        sources: List[str],
        candidates: List[str],
        configuration: str = "conditional",
        show_progress: bool = True,
    ) -> np.ndarray:
        """Compute BARTScore with specified configuration.

        Args:
            configuration: 'conditional', 'marginal', 'reverse', 'pmi'
        """
        if configuration == "conditional":
            return self._score_pairs(sources, candidates)
        elif configuration == "marginal":
            return self._score_marginal(candidates)
        elif configuration == "reverse":
            return self._score_pairs(candidates, sources)
        elif configuration == "pmi":
            cond = self._score_pairs(sources, candidates)
            mar = self._score_marginal(candidates)
            return cond - mar
        else:
            raise ValueError(f"Unknown configuration: {configuration}")

    def score_all_configurations(
        self,
        sources: List[str],
        candidates: List[str],
    ) -> Dict[str, np.ndarray]:
        """Compute all BARTScore configurations at once."""
        cond = self._score_pairs(sources, candidates)
        mar = self._score_marginal(candidates)
        rev = self._score_pairs(candidates, sources)

        return {
            "bartscore_cond": cond,
            "bartscore_mar": mar,
            "bartscore_rev": rev,
            "bartscore_pmi": cond - mar,
            "bartscore_bi": 0.5 * cond + 0.5 * rev,
        }
