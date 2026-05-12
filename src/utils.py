"""
GPU memory management utilities.

Provides safe, thorough GPU memory release for baseline models
to prevent OOM when running multiple baselines sequentially.
"""

import gc
import logging
from typing import Any

logger = logging.getLogger(__name__)


def release_model(*models_or_objects: Any) -> None:
    """Safely release one or more PyTorch models/objects from GPU memory.

    Steps:
      1. Move model to CPU (if applicable)
      2. Delete all references
      3. Run Python garbage collection
      4. Empty CUDA cache

    Args:
        *models_or_objects: model instances, scorers, or any objects holding
                           GPU tensors to be freed.
    """
    import torch

    for obj in models_or_objects:
        if obj is None:
            continue
        # If the object has a `.model` attribute (wrapper pattern), move it too
        inner = getattr(obj, "model", None)
        if inner is not None and hasattr(inner, "cpu"):
            try:
                inner.cpu()
            except Exception:
                pass
        if hasattr(obj, "cpu"):
            try:
                obj.cpu()
            except Exception:
                pass
        del obj

    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        logger.info(
            f"GPU memory after release: "
            f"{allocated:.2f} GB allocated, {reserved:.2f} GB reserved"
        )
