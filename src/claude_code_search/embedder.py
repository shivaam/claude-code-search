"""Thin wrapper around sentence-transformers for embeddings."""
from __future__ import annotations

import os

import numpy as np

# Quiet the HuggingFace/transformers ecosystem before anything imports them.
# sentence-transformers / transformers print a "BertModel LOAD REPORT" and
# weight-loading progress bars to stdout; we need stdout clean for --json.
# HF_HUB_OFFLINE=1 additionally skips the "check for updates" HTTP call on
# every load, which saves ~2 seconds per invocation once the model is cached
# and silences the "unauthenticated requests to HF Hub" warning. If the model
# is not yet cached, set CCSEARCH_HF_ONLINE=1 in your environment to allow
# the initial download.
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
if os.environ.get("CCSEARCH_HF_ONLINE") != "1":
    os.environ.setdefault("HF_HUB_OFFLINE", "1")


def _resolve_device(requested: str) -> str:
    """Return a concrete backend name for sentence-transformers.

    If the user asks for "auto", pick mps on Apple Silicon, cuda on NVIDIA,
    else cpu. Explicit values pass through unchanged.
    """
    if requested and requested != "auto":
        return requested
    try:
        import torch

        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"


class Embedder:
    def __init__(
        self, model_name: str, device: str = "auto", batch_size: int = 64
    ) -> None:
        from sentence_transformers import SentenceTransformer

        resolved = _resolve_device(device)
        try:
            self._model = SentenceTransformer(model_name, device=resolved)
        except Exception as e:  # pragma: no cover
            raise RuntimeError(
                f"Failed to load `{model_name}` on device `{resolved}`. "
                f"First run needs network to download ~130 MB into "
                f"~/.cache/huggingface/. If you are offline, set "
                f"CCSEARCH_HF_ONLINE=1 once to allow the download. "
                f"Original error: {e}"
            ) from e
        self._device = resolved
        self._batch_size = batch_size

    def embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, 384), dtype=np.float32)
        vecs = self._model.encode(
            texts,
            batch_size=self._batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return vecs.astype(np.float32, copy=False)
