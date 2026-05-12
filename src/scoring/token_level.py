"""
Token-Level DiffScore (Section 2.2.5).

Computes per-token quality scores by averaging log-probabilities across
multiple masking contexts. Enables quality heatmap visualization and
fine-grained error localization.
"""

import numpy as np
import torch
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from .diffscore import DiffScorer
from ..model.mdlm_wrapper import MDLMWrapper
from ..model.masking import RandomMasker
import logging

logger = logging.getLogger(__name__)


@dataclass
class TokenScoreResult:
    """Per-token scoring results."""
    tokens: List[str]
    scores: np.ndarray         # (L,) per-token log-prob scores
    normalized_scores: np.ndarray  # (L,) min-max normalized to [0, 1]


class TokenLevelScorer:
    """Compute per-token DiffScore for interpretability analysis."""

    def __init__(
        self,
        model: MDLMWrapper,
        K: int = 20,
        T: int = 10,
        batch_size: int = 8,
    ):
        self.model = model
        self.K = K
        self.T = T
        self.batch_size = batch_size
        self.masker = RandomMasker()

    def score_tokens(
        self,
        text: str,
        context: Optional[str] = None,
        prompt_template: Optional[Dict[str, str]] = None,
    ) -> TokenScoreResult:
        """Compute per-token quality scores.

        For marginal mode (no context): masks within the text.
        For conditional mode (with context): masks only the text part.
        """
        if context is not None and prompt_template is not None:
            full_ids, cand_start, cand_end = self.model.tokenize_pair(
                context, text, prompt_template
            )
            maskable_range = (cand_start, cand_end)
        else:
            full_ids = self.model.tokenize(text, add_special_tokens=True)
            maskable_range = None
            cand_start = 0
            cand_end = len(full_ids)

        L = full_ids.shape[0]
        device = self.model.model.device
        timesteps = [k / self.T for k in range(1, self.T + 1)]
        K_per_t = max(1, self.K // self.T)

        token_accum = np.zeros(L, dtype=np.float64)
        token_counts = np.zeros(L, dtype=np.float64)

        for t in timesteps:
            for _ in range(K_per_t):
                mask = self.masker.create_mask(full_ids, t, maskable_range)
                masked_input = full_ids.clone()
                masked_input[mask] = self.model.mask_token_id

                input_batch = masked_input.unsqueeze(0).to(device)
                mask_batch = mask.unsqueeze(0).to(device)
                orig_batch = full_ids.unsqueeze(0).to(device)

                token_lp = self.model.compute_token_logprobs(
                    input_batch, mask_batch, orig_batch
                )
                token_lp = token_lp.squeeze(0).cpu().numpy()

                mask_np = mask.numpy().astype(float)
                token_accum += token_lp
                token_counts += mask_np

        nonzero = token_counts > 0
        token_scores = np.zeros(L)
        token_scores[nonzero] = token_accum[nonzero] / token_counts[nonzero]

        # Extract candidate-region scores and tokens
        cand_scores = token_scores[cand_start:cand_end]
        cand_token_ids = full_ids[cand_start:cand_end].tolist()
        cand_tokens = [
            self.model.tokenizer.decode([tid]) for tid in cand_token_ids
        ]

        # Min-max normalize for visualization
        if cand_scores.max() > cand_scores.min():
            norm = (cand_scores - cand_scores.min()) / (
                cand_scores.max() - cand_scores.min()
            )
        else:
            norm = np.ones_like(cand_scores)

        return TokenScoreResult(
            tokens=cand_tokens,
            scores=cand_scores,
            normalized_scores=norm,
        )
