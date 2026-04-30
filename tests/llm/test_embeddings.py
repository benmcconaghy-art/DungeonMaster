"""Tests for ``app/llm/embeddings.py`` — the small, deterministic stuff.

The real local-model round-trip is exercised by the Phase 3 integration
test (slow; loads ~1.5GB of weights). These tests focus on:

  - The L2-normalisation utility.
  - The remote-backend HTTP plumbing (mocked).
  - Dimension-mismatch detection (the most likely production
    misconfiguration).
"""

from __future__ import annotations

import json
from typing import Any

import httpx
import numpy as np
import pytest

from app.llm.embeddings import (
    EmbeddingError,
    OpenAIEmbeddings,
    _l2_normalise,
)


def test_l2_normalise_shape_preserved() -> None:
    matrix = np.array([[3.0, 4.0], [1.0, 0.0], [0.0, 0.0]], dtype=np.float32)
    out = _l2_normalise(matrix)
    assert out.shape == matrix.shape
    assert out.dtype == np.float32


def test_l2_normalise_norms_are_unit() -> None:
    rng = np.random.default_rng(seed=0)
    matrix = rng.normal(size=(10, 16)).astype(np.float32)
    out = _l2_normalise(matrix)
    norms = np.linalg.norm(out, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-6), f"norms not unit: {norms}"


def test_l2_normalise_keeps_zero_vectors_zero() -> None:
    """The retrieval routine treats zero-norm queries as a degenerate
    case; the normaliser must not divide by zero."""

    matrix = np.zeros((3, 8), dtype=np.float32)
    out = _l2_normalise(matrix)
    assert np.array_equal(out, matrix)


def _mock_transport(payload: dict[str, Any], status: int = 200) -> httpx.MockTransport:
    """Build an httpx MockTransport that returns ``payload`` for any
    POST and 405 for anything else."""

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method != "POST":
            return httpx.Response(405)
        return httpx.Response(
            status,
            content=json.dumps(payload).encode(),
            headers={"content-type": "application/json"},
        )

    return httpx.MockTransport(handler)


def _embedding_response(vectors: list[list[float]]) -> dict[str, Any]:
    """Shape an OpenAI-style /v1/embeddings response."""

    return {
        "object": "list",
        "data": [
            {"object": "embedding", "index": i, "embedding": v} for i, v in enumerate(vectors)
        ],
        "model": "test",
    }


@pytest.mark.asyncio
async def test_openai_embeddings_round_trip(monkeypatch: pytest.MonkeyPatch) -> None:
    """Happy path: a 4-dim mock embedding endpoint round-trips through
    the client and comes out L2-normalised."""

    payload = _embedding_response([[3.0, 4.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]])
    transport = _mock_transport(payload)

    real_async_client = httpx.AsyncClient

    def fake_async_client(*args: Any, **kwargs: Any) -> httpx.AsyncClient:
        kwargs.pop("transport", None)
        return real_async_client(*args, transport=transport, **kwargs)

    monkeypatch.setattr("app.llm.embeddings.httpx.AsyncClient", fake_async_client)

    client = OpenAIEmbeddings(
        base_url="http://test/v1",
        model="test-model",
        expected_dim=4,
    )
    vectors = await client.embed(["a", "b"])
    assert vectors.shape == (2, 4)
    norms = np.linalg.norm(vectors, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-6)
    # The 5-12-0-0 row was 5 in 4-norm originally; after normalising
    # the first component is 0.6.
    assert vectors[0, 0] == pytest.approx(0.6, abs=1e-6)


@pytest.mark.asyncio
async def test_openai_embeddings_empty_input_returns_empty_array() -> None:
    client = OpenAIEmbeddings(base_url="http://unused/v1", model="x", expected_dim=8)
    out = await client.embed([])
    assert out.shape == (0, 8)


@pytest.mark.asyncio
async def test_openai_embeddings_dim_mismatch_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    """The single most likely production misconfiguration: switching to
    a model whose actual output dim differs from settings.embedding_dim
    must fail loudly, not write inconsistent BLOBs."""

    payload = _embedding_response([[1.0, 0.0, 0.0]])  # 3-dim
    transport = _mock_transport(payload)
    real_async_client = httpx.AsyncClient

    def fake_async_client(*args: Any, **kwargs: Any) -> httpx.AsyncClient:
        kwargs.pop("transport", None)
        return real_async_client(*args, transport=transport, **kwargs)

    monkeypatch.setattr("app.llm.embeddings.httpx.AsyncClient", fake_async_client)

    client = OpenAIEmbeddings(base_url="http://test/v1", model="x", expected_dim=8)
    with pytest.raises(EmbeddingError, match="dim mismatch"):
        await client.embed(["whatever"])


@pytest.mark.asyncio
async def test_openai_embeddings_response_shape_validation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A response missing the ``embedding`` key surfaces as a clear
    EmbeddingError, not a KeyError that escapes the boundary."""

    payload = {"object": "list", "data": [{"index": 0, "model": "x"}]}
    transport = _mock_transport(payload)
    real_async_client = httpx.AsyncClient

    def fake_async_client(*args: Any, **kwargs: Any) -> httpx.AsyncClient:
        kwargs.pop("transport", None)
        return real_async_client(*args, transport=transport, **kwargs)

    monkeypatch.setattr("app.llm.embeddings.httpx.AsyncClient", fake_async_client)

    client = OpenAIEmbeddings(base_url="http://test/v1", model="x", expected_dim=4)
    with pytest.raises(EmbeddingError, match="malformed embedding response"):
        await client.embed(["whatever"])


@pytest.mark.asyncio
async def test_openai_embeddings_row_count_mismatch(monkeypatch: pytest.MonkeyPatch) -> None:
    """If the server returns fewer rows than we asked for (e.g. silently
    truncated), the client refuses rather than zero-padding."""

    payload = _embedding_response([[1.0, 0.0, 0.0, 0.0]])  # 1 row for 2 inputs
    transport = _mock_transport(payload)
    real_async_client = httpx.AsyncClient

    def fake_async_client(*args: Any, **kwargs: Any) -> httpx.AsyncClient:
        kwargs.pop("transport", None)
        return real_async_client(*args, transport=transport, **kwargs)

    monkeypatch.setattr("app.llm.embeddings.httpx.AsyncClient", fake_async_client)

    client = OpenAIEmbeddings(base_url="http://test/v1", model="x", expected_dim=4)
    with pytest.raises(EmbeddingError, match="had 1 rows, expected 2"):
        await client.embed(["a", "b"])


@pytest.mark.asyncio
async def test_openai_embeddings_transport_error_wraps(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A network-layer failure becomes EmbeddingError, never an
    unhandled httpx exception."""

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("simulated network down")

    transport = httpx.MockTransport(handler)
    real_async_client = httpx.AsyncClient

    def fake_async_client(*args: Any, **kwargs: Any) -> httpx.AsyncClient:
        kwargs.pop("transport", None)
        return real_async_client(*args, transport=transport, **kwargs)

    monkeypatch.setattr("app.llm.embeddings.httpx.AsyncClient", fake_async_client)

    client = OpenAIEmbeddings(base_url="http://test/v1", model="x", expected_dim=4)
    with pytest.raises(EmbeddingError, match="embedding request failed"):
        await client.embed(["whatever"])
