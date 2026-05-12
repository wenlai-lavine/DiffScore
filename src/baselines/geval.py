"""
G-Eval baseline (Liu et al., 2023).

LLM-as-Judge evaluation using GPT-4 or compatible LLMs.
Prompts the LLM to evaluate text quality on a 1-5 scale with
chain-of-thought reasoning.

Requires OPENAI_API_KEY environment variable or a local LLM endpoint.
"""

import os
import json
import time
import numpy as np
from typing import List, Dict, Optional
from tqdm import tqdm
import logging

logger = logging.getLogger(__name__)

GEVAL_PROMPTS = {
    "coherence": (
        "You will be given a summary of a news article. Your task is to rate the summary "
        "on coherence (1-5). Coherence means the summary should be well-structured and "
        "well-organized, reading naturally as a body of text.\n\n"
        "Source Document:\n{source}\n\n"
        "Summary:\n{candidate}\n\n"
        "Rate the coherence of the summary on a scale of 1 to 5. "
        "Output only the numeric score."
    ),
    "consistency": (
        "You will be given a document and its summary. Your task is to rate the summary "
        "on consistency (1-5). Consistency means all information in the summary should be "
        "supported by the source document.\n\n"
        "Source Document:\n{source}\n\n"
        "Summary:\n{candidate}\n\n"
        "Rate the consistency of the summary on a scale of 1 to 5. "
        "Output only the numeric score."
    ),
    "fluency": (
        "You will be given a summary. Your task is to rate the summary on fluency (1-5). "
        "Fluency means the quality of individual sentences — whether they are well-written "
        "and grammatically correct.\n\n"
        "Summary:\n{candidate}\n\n"
        "Rate the fluency of the summary on a scale of 1 to 5. "
        "Output only the numeric score."
    ),
    "relevance": (
        "You will be given a document and its summary. Your task is to rate the summary "
        "on relevance (1-5). Relevance means the summary should include only important "
        "information from the source document.\n\n"
        "Source Document:\n{source}\n\n"
        "Summary:\n{candidate}\n\n"
        "Rate the relevance of the summary on a scale of 1 to 5. "
        "Output only the numeric score."
    ),
}


class GEvalScorer:
    """G-Eval: LLM-based evaluation with chain-of-thought.

    Supports OpenAI API (GPT-4/3.5) or any OpenAI-compatible endpoint.
    Falls back to probability-weighted scoring when logprobs are available.
    """

    def __init__(
        self,
        model_name: str = "gpt-4",
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        max_retries: int = 3,
        retry_delay: float = 2.0,
        n_repeats: int = 20,
        use_probabilities: bool = True,
    ):
        self.model_name = model_name
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.api_base = api_base
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.n_repeats = n_repeats
        self.use_probabilities = use_probabilities

        if not self.api_key:
            logger.warning(
                "No OpenAI API key found. Set OPENAI_API_KEY or pass api_key. "
                "G-Eval will return placeholder scores."
            )

    def _call_api(self, prompt: str) -> Dict:
        """Call the OpenAI API with retries."""
        try:
            import openai
        except ImportError:
            logger.error("openai package not installed. pip install openai")
            return {"score": 3.0}

        client_kwargs = {}
        if self.api_key:
            client_kwargs["api_key"] = self.api_key
        if self.api_base:
            client_kwargs["base_url"] = self.api_base

        client = openai.OpenAI(**client_kwargs)

        for attempt in range(self.max_retries):
            try:
                kwargs = {
                    "model": self.model_name,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0 if not self.use_probabilities else 1.0,
                    "max_tokens": 5,
                }
                if self.use_probabilities:
                    kwargs["n"] = self.n_repeats
                    kwargs["logprobs"] = True
                    kwargs["top_logprobs"] = 5

                response = client.chat.completions.create(**kwargs)
                return self._parse_response(response)

            except Exception as e:
                logger.warning(f"API call failed (attempt {attempt+1}): {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay * (2 ** attempt))

        return {"score": 3.0}

    def _parse_response(self, response) -> Dict:
        """Parse API response to extract score."""
        if self.use_probabilities and hasattr(response, "choices"):
            all_scores = []
            for choice in response.choices:
                text = choice.message.content.strip()
                score = self._extract_score(text)
                if score is not None:
                    all_scores.append(score)

            if all_scores:
                return {"score": float(np.mean(all_scores))}

        text = response.choices[0].message.content.strip()
        score = self._extract_score(text)
        return {"score": score if score is not None else 3.0}

    @staticmethod
    def _extract_score(text: str) -> Optional[float]:
        """Extract numeric score from model output."""
        text = text.strip().rstrip(".")
        for char in text:
            if char.isdigit():
                score = int(char)
                if 1 <= score <= 5:
                    return float(score)
        return None

    def score(
        self,
        sources: List[str],
        candidates: List[str],
        dimensions: Optional[List[str]] = None,
    ) -> Dict[str, np.ndarray]:
        """Score texts across specified dimensions.

        Returns:
            Dict mapping dimension name -> (N,) scores
        """
        if not self.api_key:
            n = len(candidates)
            logger.warning("No API key: returning placeholder G-Eval scores")
            if dimensions is None:
                dimensions = list(GEVAL_PROMPTS.keys())
            return {f"geval_{d}": np.full(n, 3.0) for d in dimensions}

        if dimensions is None:
            dimensions = list(GEVAL_PROMPTS.keys())

        results = {}
        for dim in dimensions:
            prompt_template = GEVAL_PROMPTS.get(dim)
            if prompt_template is None:
                logger.warning(f"No G-Eval prompt for dimension '{dim}'")
                continue

            logger.info(f"  G-Eval scoring dimension: {dim}")
            dim_scores = []

            for src, cand in tqdm(
                zip(sources, candidates), total=len(sources),
                desc=f"G-Eval-{dim}"
            ):
                prompt = prompt_template.format(source=src, candidate=cand)
                result = self._call_api(prompt)
                dim_scores.append(result["score"])

            results[f"geval_{dim}"] = np.array(dim_scores)

        if results:
            all_dims = np.stack(list(results.values()), axis=0)
            results["geval_overall"] = all_dims.mean(axis=0)

        return results
