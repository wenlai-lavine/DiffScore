"""
Position Bias Analysis (Section 2.3.2 / Experiment 8).

Measures how token-level scoring variance differs across positions.
DiffScore should show more uniform PosBias(n) than AR baselines.

PosBias(n) = Std_j [log p(x_n^(j) | context_n^(j))]
"""

import torch
import numpy as np
from typing import List, Dict, Optional, Tuple
from tqdm import tqdm
import logging

logger = logging.getLogger(__name__)


class PositionBiasAnalyzer:
    """Analyze position bias in scoring models."""

    def __init__(self, max_positions: int = 200):
        self.max_positions = max_positions

    def compute_diffscore_position_bias(
        self,
        model,
        texts: List[str],
        K: int = 10,
        T: int = 10,
        show_progress: bool = True,
    ) -> np.ndarray:
        """Compute per-position scoring std for DiffScore (MDLLM).

        For each text, at each position, we collect log-probs across
        multiple masking contexts, then compute std across texts at each position.

        Returns:
            pos_std: (max_positions,) std of log-probs at each position
        """
        from ..model.masking import RandomMasker

        masker = RandomMasker()
        device = model.model.device
        timesteps = [k / T for k in range(1, T + 1)]
        K_per_t = max(1, K // T)

        # Collect per-position scores: position -> list of scores across texts
        position_scores = [[] for _ in range(self.max_positions)]

        iterator = tqdm(texts, desc="PosBias-DiffScore") if show_progress else texts
        for text in iterator:
            token_ids = model.tokenize(text, add_special_tokens=True)
            L = min(len(token_ids), self.max_positions)

            # Accumulate per-token scores across MC samples
            token_accum = np.zeros(L)
            token_count = np.zeros(L)

            for t in timesteps:
                for _ in range(K_per_t):
                    mask = masker.create_mask(token_ids, t)
                    masked_input = token_ids.clone()
                    masked_input[mask] = model.mask_token_id

                    input_batch = masked_input.unsqueeze(0).to(device)
                    mask_batch = mask.unsqueeze(0).to(device)
                    orig_batch = token_ids.unsqueeze(0).to(device)

                    token_lp = model.compute_token_logprobs(
                        input_batch, mask_batch, orig_batch
                    )
                    lp = token_lp.squeeze(0).cpu().numpy()[:L]
                    m = mask.numpy()[:L].astype(float)

                    token_accum += lp
                    token_count += m

            nonzero = token_count > 0
            avg_scores = np.zeros(L)
            avg_scores[nonzero] = token_accum[nonzero] / token_count[nonzero]

            for pos in range(L):
                if nonzero[pos]:
                    position_scores[pos].append(avg_scores[pos])

        # Compute std at each position
        pos_std = np.zeros(self.max_positions)
        for pos in range(self.max_positions):
            if len(position_scores[pos]) >= 2:
                pos_std[pos] = np.std(position_scores[pos])

        return pos_std

    def compute_ar_position_bias(
        self,
        model,
        tokenizer,
        texts: List[str],
        show_progress: bool = True,
    ) -> np.ndarray:
        """Compute per-position scoring std for an autoregressive model.

        For AR models, context at position n is x^{<n}.

        Returns:
            pos_std: (max_positions,) std of log-probs at each position
        """
        import torch.nn.functional as F

        device = next(model.parameters()).device
        position_scores = [[] for _ in range(self.max_positions)]

        iterator = tqdm(texts, desc="PosBias-AR") if show_progress else texts
        for text in iterator:
            enc = tokenizer.encode(
                text, add_special_tokens=True,
                max_length=self.max_positions, truncation=True
            )
            input_ids = torch.tensor([enc], dtype=torch.long).to(device)

            with torch.no_grad():
                outputs = model(input_ids=input_ids)
                logits = outputs.logits  # (1, L, V)

            log_probs = F.log_softmax(logits, dim=-1)
            L = min(len(enc), self.max_positions)

            for pos in range(1, L):
                token_id = enc[pos]
                lp = log_probs[0, pos - 1, token_id].item()
                position_scores[pos].append(lp)

        pos_std = np.zeros(self.max_positions)
        for pos in range(self.max_positions):
            if len(position_scores[pos]) >= 2:
                pos_std[pos] = np.std(position_scores[pos])

        return pos_std

    @staticmethod
    def compute_uniformity_metric(pos_std: np.ndarray) -> float:
        """Compute uniformity of position bias (lower = more uniform).

        Returns coefficient of variation of the position std values.
        """
        valid = pos_std[pos_std > 0]
        if len(valid) < 2:
            return 0.0
        return float(np.std(valid) / (np.mean(valid) + 1e-10))
