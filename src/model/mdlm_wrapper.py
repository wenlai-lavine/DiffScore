"""
Wrapper for Masked Diffusion Language Models (LLaDA, Dream, MDLM).
Handles model loading, tokenization, and forward pass for computing
log-probabilities at masked positions.
"""

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel
from typing import Optional, List, Tuple, Dict
import logging
import os

logger = logging.getLogger(__name__)

TASK_SYSTEM_PROMPTS = {
    "summarization": "Below is a source document. The following is a summary of the document.",
    "translation": "Below is a source text and its translation.",
    "data2text": "Below is structured data. The following is a natural language description of the data.",
    "dialogue": "Below is a dialogue context and a response.",
    "default": "Below is a source text and a target text.",
}


class MDLMWrapper:
    """Unified wrapper for masked diffusion language models.

    Supports LLaDA (8B Instruct/Base), Dream, and MDLM variants.
    The key operation is: given partially masked input, compute log-probabilities
    for the original tokens at masked positions.

    For Instruct models (auto-detected via model name), applies the model's
    chat template (system/user/assistant format) in tokenize_pair to ensure
    the input matches the model's expected prompt structure.
    """

    KNOWN_MASK_TOKENS = {
        "GSAI-ML/LLaDA-8B-Instruct": 126336,
        "GSAI-ML/LLaDA-8B-Base": 126336,
        "Dream-org/Dream-v0-Instruct-7B": 151666,
        "Dream-org/Dream-v0-Base-7B": 151666,
        "kuleshov-group/mdlm-owt": None,
    }

    _AUTOMODEL_PREFIXES = ("Dream-org/",)

    _INSTRUCT_MARKERS = ("Instruct", "instruct", "chat", "Chat")

    def __init__(
        self,
        model_name: str = "GSAI-ML/LLaDA-8B-Instruct",
        device: str = "cuda",
        dtype: str = "bfloat16",
        mask_token_id: Optional[int] = None,
        max_length: int = 2048,
        adapter_path: Optional[str] = None,
        use_chat_template: Optional[bool] = None,
    ):
        self.model_name = model_name
        self.device = device
        self.max_length = max_length
        self.dtype = getattr(torch, dtype)
        self.adapter_path = adapter_path

        if use_chat_template is not None:
            self.use_chat_template = use_chat_template
        else:
            self.use_chat_template = any(
                m in model_name for m in self._INSTRUCT_MARKERS
            )

        logger.info(f"Loading model: {model_name}")
        logger.info(f"Chat template mode: {self.use_chat_template}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True
        )
        loader_cls = self._get_model_loader()
        logger.info(f"Using loader: {loader_cls.__name__}")
        base_model = loader_cls.from_pretrained(
            model_name,
            torch_dtype=self.dtype,
            trust_remote_code=True,
        )

        if adapter_path and os.path.exists(adapter_path):
            logger.info(f"Loading LoRA adapter from: {adapter_path}")
            from peft import PeftModel
            base_model = PeftModel.from_pretrained(base_model, adapter_path)
            base_model = base_model.merge_and_unload()
            logger.info("LoRA adapter merged into base model")

        self.model = base_model.to(device)
        self.model.eval()

        self.mask_token_id = mask_token_id or self._resolve_mask_token_id()
        logger.info(f"Mask token ID: {self.mask_token_id}")

        if hasattr(self.model, "config") and hasattr(self.model.config, "mask_token_id"):
            cfg_mask = self.model.config.mask_token_id
            if cfg_mask is not None and cfg_mask != self.mask_token_id:
                logger.warning(
                    f"Mask token mismatch: resolved={self.mask_token_id}, "
                    f"model.config={cfg_mask}. Using resolved value."
                )

        self._has_chat_template = (
            hasattr(self.tokenizer, "chat_template")
            and self.tokenizer.chat_template is not None
        )
        if self.use_chat_template and not self._has_chat_template:
            logger.warning(
                "use_chat_template=True but tokenizer has no chat_template. "
                "Falling back to plain text tokenization."
            )
            self.use_chat_template = False

    def _get_model_loader(self):
        """Dream models require AutoModel; LLaDA and others use AutoModelForCausalLM."""
        if any(self.model_name.startswith(p) for p in self._AUTOMODEL_PREFIXES):
            return AutoModel
        return AutoModelForCausalLM

    def _resolve_mask_token_id(self) -> int:
        if self.model_name in self.KNOWN_MASK_TOKENS:
            tid = self.KNOWN_MASK_TOKENS[self.model_name]
            if tid is not None:
                return tid

        if hasattr(self.tokenizer, "mask_token_id") and self.tokenizer.mask_token_id is not None:
            return self.tokenizer.mask_token_id

        if hasattr(self.model, "config") and hasattr(self.model.config, "mask_token_id"):
            cfg_tid = self.model.config.mask_token_id
            if cfg_tid is not None:
                logger.info(f"Resolved mask_token_id={cfg_tid} from model config")
                return cfg_tid

        for candidate in ["[MASK]", "<mask>", "<MASK>", "<|mask|>"]:
            encoded = self.tokenizer.encode(candidate, add_special_tokens=False)
            if len(encoded) == 1:
                return encoded[0]

        raise ValueError(
            f"Cannot resolve mask token for {self.model_name}. "
            "Provide mask_token_id explicitly."
        )

    @torch.no_grad()
    def _forward_logprobs(
        self,
        input_ids: torch.LongTensor,
        mask_positions: torch.BoolTensor,
        original_ids: torch.LongTensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Shared forward pass returning per-token log-probs at masked positions.

        Returns:
            target_log_probs: (B, L) log-probs at masked positions, 0 elsewhere
            n_masked: (B,) count of masked tokens per sample
        """
        outputs = self.model(input_ids=input_ids, use_cache=False)
        logits = outputs.logits  # (B, L, V)

        log_probs_all = F.log_softmax(logits, dim=-1)  # (B, L, V)

        target_log_probs = log_probs_all.gather(
            dim=-1, index=original_ids.unsqueeze(-1)
        ).squeeze(-1)  # (B, L)

        target_log_probs = target_log_probs * mask_positions.float()
        n_masked = mask_positions.float().sum(dim=-1).clamp(min=1)

        return target_log_probs, n_masked

    @torch.no_grad()
    def compute_logprobs_at_masked(
        self,
        input_ids: torch.LongTensor,
        mask_positions: torch.BoolTensor,
        original_ids: torch.LongTensor,
    ) -> torch.Tensor:
        """Compute average log-probability per sample over masked positions.

        Returns:
            avg_log_probs: (B,) average log-prob per sample
        """
        target_log_probs, n_masked = self._forward_logprobs(
            input_ids, mask_positions, original_ids
        )
        return target_log_probs.sum(dim=-1) / n_masked

    @torch.no_grad()
    def compute_sum_logprobs_at_masked(
        self,
        input_ids: torch.LongTensor,
        mask_positions: torch.BoolTensor,
        original_ids: torch.LongTensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute SUM of log-probs at masked positions (for ELBO-correct scoring).

        Returns:
            sum_log_probs: (B,) sum of log-probs per sample
            n_masked: (B,) count of masked tokens per sample
        """
        target_log_probs, n_masked = self._forward_logprobs(
            input_ids, mask_positions, original_ids
        )
        return target_log_probs.sum(dim=-1), n_masked

    @torch.no_grad()
    def compute_token_logprobs(
        self,
        input_ids: torch.LongTensor,
        mask_positions: torch.BoolTensor,
        original_ids: torch.LongTensor,
    ) -> torch.Tensor:
        """Compute per-token log-probabilities at masked positions.

        Reuses _forward_logprobs to avoid duplicate forward passes when
        called independently.

        Returns:
            token_log_probs: (B, L) log-probs at masked positions, 0 elsewhere
        """
        target_log_probs, _ = self._forward_logprobs(
            input_ids, mask_positions, original_ids
        )
        return target_log_probs

    def tokenize(
        self, text: str, add_special_tokens: bool = True
    ) -> torch.LongTensor:
        encoded = self.tokenizer.encode(
            text,
            add_special_tokens=add_special_tokens,
            max_length=self.max_length,
            truncation=True,
        )
        return torch.tensor(encoded, dtype=torch.long)

    def tokenize_pair(
        self,
        source: str,
        candidate: str,
        prompt_template: Dict[str, str],
    ) -> Tuple[torch.LongTensor, int, int]:
        """Tokenize source-candidate pair with prompt template.

        For Instruct models with a chat template, formats the input as:
            system: {task description}
            user:   {source}
            assistant: {candidate}
        and masks only the assistant (candidate) tokens.

        For base models, uses plain text concatenation:
            {prefix}{source}{mid}{candidate}{suffix}

        Returns:
            input_ids: full tokenized sequence
            cand_start: start index of candidate tokens
            cand_end: end index of candidate tokens (exclusive)
        """
        if self.use_chat_template:
            return self._tokenize_pair_chat(source, candidate, prompt_template)
        return self._tokenize_pair_plain(source, candidate, prompt_template)

    def _infer_task_from_template(self, prompt_template: Dict[str, str]) -> str:
        """Infer the task type from the prompt template for system prompt selection."""
        prefix = prompt_template.get("prefix", "").lower()
        mid = prompt_template.get("mid", "").lower()
        combined = prefix + mid

        if "summar" in combined or "document" in combined:
            return "summarization"
        if "translat" in combined or "source" in combined or "reference" in combined:
            return "translation"
        if "data" in combined:
            return "data2text"
        if "dialogue" in combined or "context" in combined or "response" in combined:
            return "dialogue"
        return "default"

    def _tokenize_pair_chat(
        self,
        source: str,
        candidate: str,
        prompt_template: Dict[str, str],
    ) -> Tuple[torch.LongTensor, int, int]:
        """Tokenize using the model's chat template.

        Constructs system/user/assistant messages and uses apply_chat_template.
        Precisely identifies candidate token boundaries by comparing tokenized
        outputs of the full conversation vs. the prefix-only conversation.
        """
        system_prompt = prompt_template.get("system", "")
        if not system_prompt:
            task = self._infer_task_from_template(prompt_template)
            system_prompt = TASK_SYSTEM_PROMPTS.get(task, TASK_SYSTEM_PROMPTS["default"])

        prefix_messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": source},
        ]

        full_messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": source},
            {"role": "assistant", "content": candidate},
        ]

        empty_messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": source},
            {"role": "assistant", "content": ""},
        ]

        try:
            prefix_ids = self.tokenizer.apply_chat_template(
                prefix_messages, tokenize=True,
                add_generation_prompt=True,
            )
            full_ids = self.tokenizer.apply_chat_template(
                full_messages, tokenize=True,
                add_generation_prompt=False,
            )
            empty_ids = self.tokenizer.apply_chat_template(
                empty_messages, tokenize=True,
                add_generation_prompt=False,
            )
        except TypeError:
            prefix_ids = self.tokenizer.apply_chat_template(
                prefix_messages, tokenize=True,
            )
            full_ids = self.tokenizer.apply_chat_template(
                full_messages, tokenize=True,
            )
            empty_ids = self.tokenizer.apply_chat_template(
                empty_messages, tokenize=True,
            )

        n_trailing = len(empty_ids) - len(prefix_ids)

        cand_start = len(prefix_ids)
        cand_end = len(full_ids) - max(0, n_trailing)

        if len(full_ids) > self.max_length:
            full_ids = full_ids[:self.max_length]
            cand_end = min(cand_end, self.max_length)

        cand_start = max(0, min(cand_start, len(full_ids)))
        cand_end = max(cand_start, min(cand_end, len(full_ids)))

        if cand_end <= cand_start:
            logger.warning(
                f"Chat template: empty candidate range "
                f"(cand_start={cand_start}, cand_end={cand_end}, "
                f"full_len={len(full_ids)}, prefix_len={len(prefix_ids)}, "
                f"empty_len={len(empty_ids)}, n_trailing={n_trailing})"
            )

        return torch.tensor(full_ids, dtype=torch.long), cand_start, cand_end

    def _tokenize_pair_plain(
        self,
        source: str,
        candidate: str,
        prompt_template: Dict[str, str],
    ) -> Tuple[torch.LongTensor, int, int]:
        """Tokenize using plain text concatenation (for base models).

        Uses character-offset alignment to robustly determine candidate
        token boundaries, avoiding subword tokenization misalignment.
        """
        prefix = prompt_template.get("prefix", "")
        mid = prompt_template.get("mid", "")
        suffix = prompt_template.get("suffix", "")

        full_text = f"{prefix}{source}{mid}{candidate}{suffix}"
        prefix_source_text = f"{prefix}{source}{mid}"

        use_offsets = getattr(self.tokenizer, "is_fast", False)
        full_enc = self.tokenizer(
            full_text,
            add_special_tokens=True,
            max_length=self.max_length,
            truncation=True,
            return_offsets_mapping=use_offsets,
        )
        full_ids = full_enc["input_ids"]
        offsets = full_enc.get("offset_mapping") if use_offsets else None

        cand_char_start = len(prefix_source_text)
        cand_char_end = len(prefix_source_text) + len(candidate)

        if offsets is not None:
            cand_start = None
            cand_end = None
            for i, (s, e) in enumerate(offsets):
                if s == 0 and e == 0:
                    continue
                if cand_start is None and e > cand_char_start:
                    cand_start = i
                if s < cand_char_end:
                    cand_end = i + 1
            if cand_start is None:
                cand_start = len(full_ids)
            if cand_end is None:
                cand_end = len(full_ids)
        else:
            prefix_source_ids = self.tokenizer.encode(
                prefix_source_text, add_special_tokens=True
            )
            cand_start = len(prefix_source_ids)
            if suffix:
                suffix_ids = self.tokenizer.encode(suffix, add_special_tokens=False)
                cand_end = len(full_ids) - len(suffix_ids)
            else:
                cand_end = len(full_ids)

        cand_start = max(0, min(cand_start, len(full_ids)))
        cand_end = max(cand_start, min(cand_end, len(full_ids)))

        return torch.tensor(full_ids, dtype=torch.long), cand_start, cand_end

    def get_vocab_size(self) -> int:
        return self.model.config.vocab_size
