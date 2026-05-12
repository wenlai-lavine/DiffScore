"""
Core DiffScore computation engine.

Implements the four basic scoring configurations (marginal, conditional,
reverse, bidirectional) with Monte Carlo estimation over stratified timesteps.

Supports three scoring modes:
  - "elbo": Faithful ELBO computation: (1/L) * E_t[(1/t) * sum_masked log p]
  - "mean_lp": Average log-prob at masked positions (no 1/t weighting)
  - "weighted": Learned or manual per-timestep weighting
"""

import torch
import numpy as np
from typing import Optional, Dict, List, Tuple, Any
from dataclasses import dataclass, field
from tqdm import tqdm
import logging

from ..model.mdlm_wrapper import MDLMWrapper
from ..model.masking import RandomMasker, BaseMasker

logger = logging.getLogger(__name__)


@dataclass
class DiffScoreResult:
    """Container for DiffScore computation results."""
    scalar: float                                     # aggregated score
    profile: Dict[float, float] = field(default_factory=dict)  # timestep -> score
    raw_profile: Dict[float, float] = field(default_factory=dict)  # without 1/t
    token_scores: Optional[np.ndarray] = None         # per-token scores
    n_forward_passes: int = 0


class DiffScorer:
    """Multi-configuration DiffScore evaluator.

    Implements all four scoring configurations from the proposal:
    (A) Marginal  - fluency evaluation
    (B) Conditional - faithfulness evaluation
    (C) Reverse   - coverage evaluation
    (D) Bidirectional - comprehensive evaluation
    """

    SCORING_MODES = ("elbo", "mean_lp", "weighted")

    def __init__(
        self,
        model: MDLMWrapper,
        K: int = 20,
        T: int = 10,
        bi_alpha: float = 0.7,
        masker: Optional[BaseMasker] = None,
        normalize: str = "length",
        batch_size: int = 8,
        stratified: bool = True,
        scoring_mode: str = "mean_lp",
        timestep_weights: Optional[np.ndarray] = None,
        min_t: float = 0.05,
        exclude_t_1: bool = True,
    ):
        self.model = model
        self.K = K
        self.T = T
        self.bi_alpha = bi_alpha
        self.masker = masker or RandomMasker()
        self.normalize = normalize
        self.batch_size = batch_size
        self.stratified = stratified
        self.scoring_mode = scoring_mode
        self.timestep_weights = timestep_weights
        self.min_t = min_t
        self.exclude_t_1 = exclude_t_1

        if scoring_mode not in self.SCORING_MODES:
            raise ValueError(
                f"scoring_mode must be one of {self.SCORING_MODES}, got '{scoring_mode}'"
            )
        if scoring_mode == "weighted" and timestep_weights is None:
            logger.warning("weighted mode without weights; falling back to uniform")

    def _get_timesteps(self) -> List[float]:
        """Generate stratified timesteps, clamped to [min_t, 1.0].

        When exclude_t_1 is True, the t=1.0 timestep (full masking) is
        excluded since it provides no useful context signal -- all tokens
        are masked and the model score degenerates.
        """
        steps = [max(self.min_t, k / self.T) for k in range(1, self.T + 1)]
        if self.exclude_t_1:
            steps = [t for t in steps if t < 1.0]
        return steps

    def _apply_mask(
        self,
        original_ids: torch.LongTensor,
        mask: torch.BoolTensor,
    ) -> torch.LongTensor:
        """Replace masked positions with [MASK] token id."""
        masked = original_ids.clone()
        masked[mask] = self.model.mask_token_id
        return masked

    def _score_single_timestep(
        self,
        original_ids: torch.LongTensor,
        t: float,
        K_per_t: int,
        maskable_range: Optional[Tuple[int, int]] = None,
        return_token_scores: bool = False,
    ) -> Tuple[float, float, Optional[np.ndarray]]:
        """Compute score at a single timestep with K_per_t MC samples.

        Returns:
            sum_score: average across samples of (sum of log-probs / L_maskable)
            avg_score: average across samples of (mean log-prob at masked positions)
            token_avg: optional per-token average scores
        """
        L = original_ids.shape[0]
        device = self.model.model.device

        start = maskable_range[0] if maskable_range else 0
        end = maskable_range[1] if maskable_range else L
        L_maskable = max(1, end - start)

        all_sum_scores = []
        all_avg_scores = []
        token_accum = np.zeros(L, dtype=np.float64) if return_token_scores else None
        token_counts = np.zeros(L, dtype=np.float64) if return_token_scores else None

        for batch_start in range(0, K_per_t, self.batch_size):
            batch_end = min(batch_start + self.batch_size, K_per_t)
            n_batch = batch_end - batch_start

            if isinstance(self.masker, RandomMasker):
                masks = self.masker.create_batch_masks(
                    original_ids, t, n_batch, maskable_range
                )
            else:
                masks = torch.stack([
                    self.masker.create_mask(original_ids, t, maskable_range)
                    for _ in range(n_batch)
                ])

            input_batch = original_ids.unsqueeze(0).expand(n_batch, -1).clone()
            input_batch[masks] = self.model.mask_token_id

            original_batch = original_ids.unsqueeze(0).expand(n_batch, -1)

            input_batch = input_batch.to(device)
            masks = masks.to(device)
            original_batch = original_batch.to(device)

            sum_lp, n_masked = self.model.compute_sum_logprobs_at_masked(
                input_batch, masks, original_batch
            )

            sum_scores = (sum_lp / L_maskable).cpu().tolist()
            avg_scores = (sum_lp / n_masked).cpu().tolist()
            all_sum_scores.extend(sum_scores)
            all_avg_scores.extend(avg_scores)

            if return_token_scores:
                token_lp = self.model.compute_token_logprobs(
                    input_batch, masks, original_batch
                )
                token_accum += token_lp.cpu().numpy().sum(axis=0)
                token_counts += masks.cpu().numpy().sum(axis=0)

        mean_sum = np.mean(all_sum_scores)
        mean_avg = np.mean(all_avg_scores)

        token_avg = None
        if return_token_scores:
            nonzero = token_counts > 0
            token_avg = np.zeros(L)
            token_avg[nonzero] = token_accum[nonzero] / token_counts[nonzero]

        return mean_sum, mean_avg, token_avg

    def score_marginal(
        self,
        text: str,
        return_profile: bool = True,
        return_token_scores: bool = False,
    ) -> DiffScoreResult:
        """Configuration A: Marginal scoring for fluency evaluation."""
        original_ids = self.model.tokenize(text, add_special_tokens=True)
        return self._score_text(
            original_ids, maskable_range=None,
            return_profile=return_profile,
            return_token_scores=return_token_scores,
        )

    def score_conditional(
        self,
        source: str,
        candidate: str,
        prompt_template: Dict[str, str],
        return_profile: bool = True,
        return_token_scores: bool = False,
    ) -> DiffScoreResult:
        """Configuration B: Conditional scoring for faithfulness evaluation."""
        full_ids, cand_start, cand_end = self.model.tokenize_pair(
            source, candidate, prompt_template
        )
        return self._score_text(
            full_ids, maskable_range=(cand_start, cand_end),
            return_profile=return_profile,
            return_token_scores=return_token_scores,
        )

    def score_reverse(
        self,
        source: str,
        candidate: str,
        prompt_template: Dict[str, str],
        return_profile: bool = True,
        return_token_scores: bool = False,
    ) -> DiffScoreResult:
        """Configuration C: Reverse scoring for coverage evaluation.

        Uses a dedicated reverse template for natural prompt ordering.
        """
        rev_template = self._build_reverse_template(prompt_template)
        full_ids, src_start, src_end = self.model.tokenize_pair(
            candidate, source, rev_template
        )
        return self._score_text(
            full_ids, maskable_range=(src_start, src_end),
            return_profile=return_profile,
            return_token_scores=return_token_scores,
        )

    def _build_reverse_template(self, prompt_template: Dict[str, str]) -> Dict[str, str]:
        """Build reverse template that reads naturally.

        For summarization: "Summary: {c}\n\nOriginal document: {s}"
        """
        prefix = prompt_template.get("prefix", "").strip()
        mid = prompt_template.get("mid", "").strip()
        suffix = prompt_template.get("suffix", "")

        REVERSE_MAPPINGS = {
            "Document:": ("Summary: ", "\n\nOriginal document: "),
            "Given the following document:": ("Given the following summary:\n", "\n\nThe original document is:\n"),
            "Article:": ("Summary: ", "\n\nOriginal article: "),
            "Source document:": ("Generated summary: ", "\nSource document: "),
            "Source": ("Translation: ", "\n\nSource: "),
            "Context:": ("Response: ", "\n\nContext: "),
            "Data:": ("Text: ", "\n\nData: "),
        }

        for key, (new_prefix, new_mid) in REVERSE_MAPPINGS.items():
            if prefix.startswith(key):
                return {"prefix": new_prefix, "mid": new_mid, "suffix": suffix}

        return {
            "prefix": mid.rstrip() + " " if mid else "Target: ",
            "mid": "\n\n" + prefix.rstrip() + " " if prefix else "\n\nSource: ",
            "suffix": suffix,
        }

    def score_bidirectional(
        self,
        source: str,
        candidate: str,
        prompt_template: Dict[str, str],
        return_profile: bool = True,
    ) -> DiffScoreResult:
        """Configuration D: Bidirectional scoring (alpha-weighted combination)."""
        cond = self.score_conditional(source, candidate, prompt_template, return_profile)
        rev = self.score_reverse(source, candidate, prompt_template, return_profile)

        scalar = self.bi_alpha * cond.scalar + (1 - self.bi_alpha) * rev.scalar
        profile = {}
        raw_profile = {}
        if return_profile:
            for t in cond.profile:
                profile[t] = (
                    self.bi_alpha * cond.profile[t]
                    + (1 - self.bi_alpha) * rev.profile.get(t, 0.0)
                )
                raw_profile[t] = (
                    self.bi_alpha * cond.raw_profile.get(t, 0.0)
                    + (1 - self.bi_alpha) * rev.raw_profile.get(t, 0.0)
                )

        return DiffScoreResult(
            scalar=scalar,
            profile=profile,
            raw_profile=raw_profile,
            n_forward_passes=cond.n_forward_passes + rev.n_forward_passes,
        )

    def _score_text(
        self,
        original_ids: torch.LongTensor,
        maskable_range: Optional[Tuple[int, int]],
        return_profile: bool = True,
        return_token_scores: bool = False,
    ) -> DiffScoreResult:
        """Core scoring logic shared across configurations.

        Scoring modes:
          - "elbo": (1/t) * (sum_log_p / L) per timestep, then average
          - "mean_lp": avg log-prob at masked positions per timestep, then average
          - "weighted": raw profile with learned/manual weights
        """
        timesteps = self._get_timesteps()
        K_per_t = max(1, self.K // self.T)
        n_forward = 0

        elbo_profile = {}
        raw_profile = {}
        L = original_ids.shape[0]
        token_accum = np.zeros(L) if return_token_scores else None
        token_weight = np.zeros(L) if return_token_scores else None

        for t in timesteps:
            sum_score, avg_score, token_scores_t = self._score_single_timestep(
                original_ids, t, K_per_t,
                maskable_range=maskable_range,
                return_token_scores=return_token_scores,
            )

            # ELBO-correct: (1/t) * E[sum_log_p / L_maskable]
            elbo_profile[t] = sum_score / t
            # Raw average log-prob at masked positions (no 1/t)
            raw_profile[t] = avg_score
            n_forward += K_per_t

            if return_token_scores and token_scores_t is not None:
                token_accum += token_scores_t
                token_weight += np.ones(L)

        if self.scoring_mode == "elbo":
            profile = elbo_profile
        elif self.scoring_mode == "mean_lp":
            profile = raw_profile
        elif self.scoring_mode == "weighted":
            profile = raw_profile
        else:
            profile = elbo_profile

        if self.scoring_mode == "weighted" and self.timestep_weights is not None:
            w = self.timestep_weights / self.timestep_weights.sum()
            vals = np.array([profile[t] for t in timesteps])
            scalar = float(np.dot(vals, w))
        else:
            scalar = float(np.mean(list(profile.values())))

        final_token_scores = None
        if return_token_scores:
            nonzero = token_weight > 0
            final_token_scores = np.zeros(L)
            final_token_scores[nonzero] = token_accum[nonzero] / token_weight[nonzero]

        return DiffScoreResult(
            scalar=scalar,
            profile=profile,
            raw_profile=raw_profile,
            token_scores=final_token_scores,
            n_forward_passes=n_forward,
        )

    def score_batch(
        self,
        texts: Optional[List[str]] = None,
        sources: Optional[List[str]] = None,
        candidates: Optional[List[str]] = None,
        prompt_template: Optional[Dict[str, str]] = None,
        configuration: str = "marginal",
        show_progress: bool = True,
    ) -> List[DiffScoreResult]:
        """Score a batch of texts/pairs with the specified configuration."""
        results = []
        if configuration == "marginal":
            assert texts is not None
            iterator = tqdm(texts, desc="DiffScore-mar") if show_progress else texts
            for text in iterator:
                results.append(self.score_marginal(text))
        elif configuration == "conditional":
            assert sources is not None and candidates is not None
            assert prompt_template is not None
            pairs = zip(sources, candidates)
            if show_progress:
                pairs = tqdm(
                    list(pairs), desc="DiffScore-cond", total=len(sources)
                )
            for src, cand in pairs:
                results.append(
                    self.score_conditional(src, cand, prompt_template)
                )
        elif configuration == "reverse":
            assert sources is not None and candidates is not None
            assert prompt_template is not None
            pairs = zip(sources, candidates)
            if show_progress:
                pairs = tqdm(
                    list(pairs), desc="DiffScore-rev", total=len(sources)
                )
            for src, cand in pairs:
                results.append(
                    self.score_reverse(src, cand, prompt_template)
                )
        elif configuration == "bidirectional":
            assert sources is not None and candidates is not None
            assert prompt_template is not None
            pairs = zip(sources, candidates)
            if show_progress:
                pairs = tqdm(
                    list(pairs), desc="DiffScore-bi", total=len(sources)
                )
            for src, cand in pairs:
                results.append(
                    self.score_bidirectional(src, cand, prompt_template)
                )
        else:
            raise ValueError(f"Unknown configuration: {configuration}")

        return results
