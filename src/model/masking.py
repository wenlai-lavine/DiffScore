"""
Masking strategies for DiffScore.

Implements random masking (core) and linguistically-driven structured masking
(Entity-Mask, Function-Mask, Content-Mask) for diagnostic analysis.
"""

import torch
import numpy as np
from typing import List, Optional, Tuple, Set
from abc import ABC, abstractmethod
import logging

logger = logging.getLogger(__name__)


class BaseMasker(ABC):
    """Base class for masking strategies."""

    @abstractmethod
    def create_mask(
        self,
        token_ids: torch.LongTensor,
        mask_ratio: float,
        maskable_range: Optional[Tuple[int, int]] = None,
    ) -> torch.BoolTensor:
        """Create a boolean mask indicating which positions to mask.

        Args:
            token_ids: (L,) original token ids
            mask_ratio: fraction of maskable tokens to mask (t in [0, 1])
            maskable_range: (start, end) indices of the maskable region.
                           If None, all positions are maskable.

        Returns:
            mask: (L,) boolean tensor, True at positions to be masked
        """
        raise NotImplementedError


class RandomMasker(BaseMasker):
    """Standard random masking as defined in MDLLM forward process.

    Each maskable token is independently masked with probability t.
    """

    def create_mask(
        self,
        token_ids: torch.LongTensor,
        mask_ratio: float,
        maskable_range: Optional[Tuple[int, int]] = None,
    ) -> torch.BoolTensor:
        L = token_ids.shape[0]
        mask = torch.zeros(L, dtype=torch.bool)

        start = maskable_range[0] if maskable_range else 0
        end = maskable_range[1] if maskable_range else L

        n_maskable = end - start
        if n_maskable <= 0:
            return mask

        rand = torch.rand(n_maskable)
        mask[start:end] = rand < mask_ratio

        # Ensure at least one token is masked
        if mask.sum() == 0 and n_maskable > 0:
            idx = torch.randint(start, end, (1,))
            mask[idx] = True

        return mask

    def create_batch_masks(
        self,
        token_ids: torch.LongTensor,
        mask_ratio: float,
        n_masks: int,
        maskable_range: Optional[Tuple[int, int]] = None,
    ) -> torch.BoolTensor:
        """Create multiple independent masks for batch processing.

        Returns:
            masks: (n_masks, L) boolean tensor
        """
        L = token_ids.shape[0]
        masks = torch.zeros(n_masks, L, dtype=torch.bool)

        start = maskable_range[0] if maskable_range else 0
        end = maskable_range[1] if maskable_range else L
        n_maskable = end - start

        if n_maskable <= 0:
            return masks

        rand = torch.rand(n_masks, n_maskable)
        masks[:, start:end] = rand < mask_ratio

        # Ensure at least one token is masked per sample
        zero_mask_rows = masks.sum(dim=1) == 0
        if zero_mask_rows.any():
            for row_idx in zero_mask_rows.nonzero(as_tuple=True)[0]:
                idx = torch.randint(start, end, (1,))
                masks[row_idx, idx] = True

        return masks


class _SpaCyMasker(BaseMasker):
    """Base class for spaCy-based structured masking."""

    def __init__(self, tokenizer, spacy_model: str = "en_core_web_sm"):
        self.tokenizer = tokenizer
        try:
            import spacy
            self.nlp = spacy.load(spacy_model)
        except OSError:
            logger.warning(
                f"spaCy model '{spacy_model}' not found. "
                f"Run: python -m spacy download {spacy_model}"
            )
            raise

    def _get_token_spans(
        self, text: str, token_ids: torch.LongTensor
    ) -> List[Tuple[int, int]]:
        """Map each token_id position back to character spans in text."""
        spans = []
        offset = 0
        for i, tid in enumerate(token_ids):
            decoded = self.tokenizer.decode([tid.item()])
            start = text.find(decoded, offset)
            if start == -1:
                start = offset
            end = start + len(decoded)
            spans.append((start, end))
            offset = max(offset, end)
        return spans

    def _get_target_char_positions(self, text: str) -> Set[int]:
        """Subclasses implement this to specify which character positions to mask."""
        raise NotImplementedError

    def create_mask(
        self,
        token_ids: torch.LongTensor,
        mask_ratio: float,
        maskable_range: Optional[Tuple[int, int]] = None,
        text: Optional[str] = None,
    ) -> torch.BoolTensor:
        if text is None:
            text = self.tokenizer.decode(token_ids.tolist())

        L = token_ids.shape[0]
        mask = torch.zeros(L, dtype=torch.bool)

        start = maskable_range[0] if maskable_range else 0
        end = maskable_range[1] if maskable_range else L

        target_chars = self._get_target_char_positions(text)
        if not target_chars:
            # Fall back to random masking
            return RandomMasker().create_mask(token_ids, mask_ratio, maskable_range)

        token_spans = self._get_token_spans(text, token_ids)

        eligible_positions = []
        for i in range(start, end):
            if i < len(token_spans):
                span_start, span_end = token_spans[i]
                if any(c in target_chars for c in range(span_start, span_end)):
                    eligible_positions.append(i)

        if not eligible_positions:
            return RandomMasker().create_mask(token_ids, mask_ratio, maskable_range)

        n_to_mask = max(1, int(len(eligible_positions) * mask_ratio))
        chosen = np.random.choice(eligible_positions, size=n_to_mask, replace=False)
        mask[chosen] = True

        return mask


class EntityMasker(_SpaCyMasker):
    """Mask named entities to probe factual consistency."""

    def _get_target_char_positions(self, text: str) -> Set[int]:
        doc = self.nlp(text)
        positions = set()
        for ent in doc.ents:
            positions.update(range(ent.start_char, ent.end_char))
        return positions


class FunctionMasker(_SpaCyMasker):
    """Mask function words (determiners, prepositions, etc.) to probe grammar."""

    FUNCTION_POS = {"DET", "ADP", "CCONJ", "SCONJ", "PART", "AUX", "PRON"}

    def _get_target_char_positions(self, text: str) -> Set[int]:
        doc = self.nlp(text)
        positions = set()
        for token in doc:
            if token.pos_ in self.FUNCTION_POS:
                positions.update(range(token.idx, token.idx + len(token.text)))
        return positions


class ContentMasker(_SpaCyMasker):
    """Mask content words (nouns, verbs, adjectives) to probe information adequacy."""

    CONTENT_POS = {"NOUN", "VERB", "ADJ", "ADV", "PROPN"}

    def _get_target_char_positions(self, text: str) -> Set[int]:
        doc = self.nlp(text)
        positions = set()
        for token in doc:
            if token.pos_ in self.CONTENT_POS:
                positions.update(range(token.idx, token.idx + len(token.text)))
        return positions
