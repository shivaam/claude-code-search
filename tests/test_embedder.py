import os

import numpy as np
import pytest

from claude_code_search.embedder import Embedder, _resolve_device


def test_resolve_device_passes_through_explicit_values() -> None:
    assert _resolve_device("cpu") == "cpu"
    assert _resolve_device("mps") == "mps"
    assert _resolve_device("cuda") == "cuda"


def test_resolve_device_auto_returns_a_concrete_backend() -> None:
    result = _resolve_device("auto")
    assert result in {"mps", "cuda", "cpu"}


@pytest.mark.skipif(
    os.environ.get("CCSEARCH_SKIP_SLOW") == "1",
    reason="requires model download",
)
def test_embedder_returns_normalized_vectors() -> None:
    emb = Embedder(model_name="BAAI/bge-small-en-v1.5", device="cpu", batch_size=8)
    out = emb.embed(["hello", "world"])
    assert out.shape == (2, 384)
    assert out.dtype == np.float32
    norms = np.linalg.norm(out, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-4)
