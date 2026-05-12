"""
MoverScore baseline (Zhao et al., 2019).

Uses Word Mover's Distance with contextualized BERT embeddings.
Computes the optimal transport distance between candidate and reference
using contextual word embeddings.
"""

import numpy as np
from typing import List, Dict, Optional, Union
import logging
import torch
import random

logger = logging.getLogger(__name__)


class MoverScorer:
    """MoverScore: Word Mover's Distance with BERT embeddings.

    Falls back to a cosine-similarity approach if the moverscore package
    is not available.
    """

    def __init__(
        self,
        model_name: str = "distilbert-base-uncased",
        device: str = "cuda",
        n_gram: int = 1,
    ):
        self.device = device
        self.model_name = model_name
        self.n_gram = n_gram
        self._has_moverscore = False

        try:
            from moverscore_v2 import get_idf_dict, word_mover_score
            self._has_moverscore = True
            self._word_mover_score = word_mover_score
            self._get_idf_dict = get_idf_dict
            logger.info("Using moverscore_v2 package")
        except ImportError:
            logger.warning(
                "moverscore_v2 not installed. Using fallback WMD implementation. "
                "Install via: pip install moverscore"
            )
            self._setup_fallback()

    def _setup_fallback(self):
        """Set up fallback using sentence-transformers for WMD approximation."""
        import torch
        from transformers import AutoModel, AutoTokenizer

        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self._model = AutoModel.from_pretrained(self.model_name).to(self.device)
        self._model.eval()

    def score(
        self,
        candidates: List[str],
        references: List[str],
    ) -> np.ndarray:
        """Compute MoverScore for each candidate-reference pair."""
        if self._has_moverscore:
            return self._score_moverscore(candidates, references)
        return self._score_fallback(candidates, references)

    def _score_moverscore(
        self, candidates: List[str], references: Union[List[str], List[List[str]]]
    ) -> np.ndarray:
        """Score using the official moverscore package."""
        idf_hyps = self._get_idf_dict(candidates)
        # moverscore does not support multi reference evaluation, use the first reference
        if isinstance(references[0], List):
            random_references = [ref[random.randint(0, len(ref) - 1)] for ref in references]
            idf_refs = self._get_idf_dict(random_references)
            scores = self._word_mover_score(
                random_references, candidates,
                idf_dict_ref=idf_refs,
                idf_dict_hyp=idf_hyps,
                stop_words=[],
                n_gram=self.n_gram,
                remove_subwords=True,
                batch_size=32,
                device=self.device,
            )
        else:
            idf_refs = self._get_idf_dict(references)
            scores = self._word_mover_score(
                references, candidates,
                idf_dict_ref=idf_refs,
                idf_dict_hyp=idf_hyps,
                stop_words=[],
                n_gram=self.n_gram,
                remove_subwords=True,
                batch_size=32,
                device=self.device,
            )
        
        return np.array(scores)

    def _score_fallback(
        self, candidates: List[str], references: List[str]
    ) -> np.ndarray:
        """Fallback: token-level cosine WMD approximation."""
        import torch

        scores = []
        for cand, ref in zip(candidates, references):
            cand_emb = self._get_token_embeddings(cand)
            ref_emb = self._get_token_embeddings(ref)

            if cand_emb is None or ref_emb is None:
                scores.append(0.0)
                continue

            # Cosine similarity matrix
            cand_norm = cand_emb / cand_emb.norm(dim=-1, keepdim=True).clamp(min=1e-8)
            ref_norm = ref_emb / ref_emb.norm(dim=-1, keepdim=True).clamp(min=1e-8)
            sim_matrix = torch.mm(cand_norm, ref_norm.t())

            # Greedy matching (fast WMD approximation)
            r_to_c = sim_matrix.max(dim=0).values.mean()
            c_to_r = sim_matrix.max(dim=1).values.mean()
            scores.append(((r_to_c + c_to_r) / 2).item())

        return np.array(scores)

    @torch.no_grad()
    def _get_token_embeddings(self, text: str):
        """Get token-level contextual embeddings."""
        import torch

        inputs = self._tokenizer(
            text, return_tensors="pt", truncation=True, max_length=512
        ).to(self.device)

        if inputs["input_ids"].shape[1] < 2:
            return None

        outputs = self._model(**inputs)
        # Skip [CLS] and [SEP]
        embeddings = outputs.last_hidden_state[0, 1:-1, :]
        return embeddings


# Make torch importable at module level for the decorator
try:
    import torch
except ImportError:
    pass
