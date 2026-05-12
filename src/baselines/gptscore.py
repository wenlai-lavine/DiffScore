"""
GPTScore baseline (Fu et al., 2023).

Computes evaluation scores using auto-regressive conditional log-probabilities
from decoder-only language models. Supports multiple configurations matching
the DiffScore comparison framework.
"""

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List, Dict, Optional
import numpy as np
from tqdm import tqdm
import logging

logger = logging.getLogger(__name__)

GPTSCORE_PROMPT_TEMPLATES = {
    "summarization": {
        "cond": "Summarize the following document.\n\nDocument: {source}\n\nSummary: {candidate}",
        "mar": "{candidate}",
    },
    "translation": {
        "cond": "Translate the following text.\n\nSource: {source}\n\nTranslation: {candidate}",
        "mar": "{candidate}",
    },
    "dialogue": {
        "cond": "Generate a response for the conversation.\n\nContext: {source}\n\nResponse: {candidate}",
        "mar": "{candidate}",
    },
    "data2text": {
        "cond": "Generate a description for the data.\n\nData: {source}\n\nDescription: {candidate}",
        "mar": "{candidate}",
    },
}


class GPTScorer:
    """GPTScore: auto-regressive log-probability evaluation.

    Uses a decoder-only LM to compute P(candidate | instruction + source)
    via teacher-forced log-probability.
    """

    def __init__(
        self,
        model_name: str = "gpt2-large",
        device: str = "cuda",
        max_length: int = 1024,
        batch_size: int = 8,
        task: str = "summarization",
    ):
        self.device = device
        self.max_length = max_length
        self.batch_size = batch_size
        self.task = task

        logger.info(f"Loading GPTScore model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.model.eval()
        self.model.to(device)

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    @torch.no_grad()
    def _score_conditional(
        self, sources: List[str], candidates: List[str]
    ) -> np.ndarray:
        """Compute log P(candidate | prompt + source) with AR model."""
        template = GPTSCORE_PROMPT_TEMPLATES.get(self.task, GPTSCORE_PROMPT_TEMPLATES["summarization"])
        scores = []

        for i in range(0, len(sources), self.batch_size):
            batch_src = sources[i:i + self.batch_size]
            batch_cand = candidates[i:i + self.batch_size]
            batch_scores = []

            for src, cand in zip(batch_src, batch_cand):
                full_text = template["cond"].format(source=src, candidate=cand)
                prefix_text = template["cond"].format(source=src, candidate="")

                full_ids = self.tokenizer.encode(
                    full_text, truncation=True, max_length=self.max_length,
                    return_tensors="pt"
                ).to(self.device)
                prefix_ids = self.tokenizer.encode(
                    prefix_text, truncation=True, max_length=self.max_length
                )
                prefix_len = len(prefix_ids)

                outputs = self.model(input_ids=full_ids)
                logits = outputs.logits  # (1, L, V)

                # Shift: predict token[i+1] from position[i]
                shift_logits = logits[:, :-1, :]
                shift_labels = full_ids[:, 1:]

                log_probs = F.log_softmax(shift_logits, dim=-1)
                token_log_probs = log_probs.gather(
                    dim=-1, index=shift_labels.unsqueeze(-1)
                ).squeeze(-1)  # (1, L-1)

                # Only score the candidate portion (after prefix)
                cand_start = max(0, prefix_len - 1)
                cand_log_probs = token_log_probs[0, cand_start:]

                if cand_log_probs.numel() == 0:
                    batch_scores.append(0.0)
                else:
                    batch_scores.append(
                        cand_log_probs.mean().item()
                    )

            scores.extend(batch_scores)

        return np.array(scores)

    @torch.no_grad()
    def _score_marginal(self, texts: List[str]) -> np.ndarray:
        """Compute marginal log P(text) with AR model."""
        scores = []

        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]

            for text in batch:
                input_ids = self.tokenizer.encode(
                    text, truncation=True, max_length=self.max_length,
                    return_tensors="pt"
                ).to(self.device)

                if input_ids.shape[1] < 2:
                    scores.append(0.0)
                    continue

                outputs = self.model(input_ids=input_ids)
                logits = outputs.logits

                shift_logits = logits[:, :-1, :]
                shift_labels = input_ids[:, 1:]

                log_probs = F.log_softmax(shift_logits, dim=-1)
                token_log_probs = log_probs.gather(
                    dim=-1, index=shift_labels.unsqueeze(-1)
                ).squeeze(-1)

                scores.append(token_log_probs.mean().item())

        return np.array(scores)

    def score(
        self,
        sources: List[str],
        candidates: List[str],
        configuration: str = "conditional",
    ) -> np.ndarray:
        if configuration == "conditional":
            return self._score_conditional(sources, candidates)
        elif configuration == "marginal":
            return self._score_marginal(candidates)
        elif configuration == "reverse":
            return self._score_conditional(candidates, sources)
        elif configuration == "pmi":
            cond = self._score_conditional(sources, candidates)
            mar = self._score_marginal(candidates)
            return cond - mar
        else:
            raise ValueError(f"Unknown configuration: {configuration}")

    def score_all_configurations(
        self,
        sources: List[str],
        candidates: List[str],
    ) -> Dict[str, np.ndarray]:
        cond = self._score_conditional(sources, candidates)
        mar = self._score_marginal(candidates)

        return {
            "gptscore_cond": cond,
            "gptscore_mar": mar,
            "gptscore_pmi": cond - mar,
        }
