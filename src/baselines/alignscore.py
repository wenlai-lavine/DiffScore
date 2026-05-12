"""
AlignScore baseline (Zha et al., 2023).

NLI-based evaluation that measures alignment between source and candidate text.
Uses a RoBERTa model fine-tuned on NLI + fact verification + paraphrase detection.
"""

import os
import torch
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from typing import List, Dict, Optional
import numpy as np
from tqdm import tqdm
import logging

logger = logging.getLogger(__name__)

DEFAULT_CKPT_PATH = os.environ.get(
    "ALIGNSCORE_CKPT",
    "AlignScore-base.ckpt",
)


class AlignScorer:
    """AlignScore: NLI-based alignment evaluation.

    Uses a model fine-tuned for textual alignment to measure how well
    a candidate text aligns with a source document.

    The official AlignScore library is used when available. Set the
    ALIGNSCORE_CKPT environment variable to point to the checkpoint file,
    or pass ckpt_path explicitly.
    """

    def __init__(
        self,
        model_name: str = "roberta-base",
        device: str = "cuda",
        max_length: int = 512,
        batch_size: int = 16,
        ckpt_path: Optional[str] = "/home/ubuntu-1/shenyingli/public_models/AlignScore/AlignScore-base.ckpt",
        evaluation_mode: str = "nli_sp",
    ):
        self.device = device
        self.max_length = max_length
        self.batch_size = batch_size
        self._using_library = False

        resolved_ckpt = ckpt_path or DEFAULT_CKPT_PATH

        try:
            from alignscore import AlignScore
            logger.info(f"Loading AlignScore via library (ckpt: {resolved_ckpt})")
            self.model = AlignScore(
                model=model_name,
                batch_size=batch_size,
                device=device,
                ckpt_path=resolved_ckpt,
                evaluation_mode=evaluation_mode,
            )
            self._using_library = True
        except (ImportError, FileNotFoundError, Exception) as e:
            logger.warning(
                f"AlignScore library not available ({e}). "
                "Falling back to NLI-based scoring with "
                "AutoModelForSequenceClassification."
            )
            hf_model = model_name if "/" in model_name else "yzha/AlignScore-large"
            self.tokenizer = AutoTokenizer.from_pretrained(hf_model)
            self.model = AutoModelForSequenceClassification.from_pretrained(hf_model)
            self.model.eval()
            self.model.to(device)
            self._n_labels = self.model.config.num_labels

    @torch.no_grad()
    def _score_pairs_nli(
        self,
        premises: List[str],
        hypotheses: List[str],
    ) -> np.ndarray:
        """Score using NLI entailment probability."""
        scores = []

        for i in range(0, len(premises), self.batch_size):
            batch_prem = premises[i:i + self.batch_size]
            batch_hyp = hypotheses[i:i + self.batch_size]

            encoded = self.tokenizer(
                batch_prem, batch_hyp,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_length,
            ).to(self.device)

            outputs = self.model(**encoded)
            logits = outputs.logits

            if self._n_labels == 3:
                # NLI-style: [contradiction, neutral, entailment]
                probs = F.softmax(logits, dim=-1)
                entailment_scores = probs[:, 2].cpu().tolist()
            elif self._n_labels == 2:
                # Binary: [not_aligned, aligned]
                probs = F.softmax(logits, dim=-1)
                entailment_scores = probs[:, 1].cpu().tolist()
            else:
                # Single logit
                entailment_scores = torch.sigmoid(logits[:, 0]).cpu().tolist()

            scores.extend(entailment_scores)

        return np.array(scores)

    def _chunk_text(self, text: str, max_words: int = 200) -> List[str]:
        """Split long text into overlapping chunks for NLI processing."""
        words = text.split()
        if len(words) <= max_words:
            return [text]

        chunks = []
        stride = max_words // 2
        for start in range(0, len(words), stride):
            chunk = " ".join(words[start:start + max_words])
            chunks.append(chunk)
            if start + max_words >= len(words):
                break
        return chunks

    def score(
        self,
        sources: List[str],
        candidates: List[str],
        configuration: str = "nli",
    ) -> np.ndarray:
        """Score alignment between sources and candidates."""
        if self._using_library:
            return np.array(
                self.model.score(contexts=sources, claims=candidates)
            )

        scores = []
        for src, cand in tqdm(
            zip(sources, candidates), total=len(sources), desc="AlignScore"
        ):
            src_chunks = self._chunk_text(src)
            chunk_scores = self._score_pairs_nli(
                src_chunks, [cand] * len(src_chunks)
            )
            scores.append(float(chunk_scores.max()))

        return np.array(scores)

    def score_all_configurations(
        self,
        sources: List[str],
        candidates: List[str],
    ) -> np.ndarray:
        """Compute AlignScore."""
        return self.score(sources, candidates)