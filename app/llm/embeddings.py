"""Embedding client.

Two backends behind a common protocol:

  - :class:`OpenAIEmbeddings` calls an OpenAI-compatible
    ``POST /v1/embeddings`` endpoint. Works against Ollama (the
    ``YOUR_AI_SERVER:11436`` deployment), vLLM-with-embeddings-enabled,
    and OpenAI itself unchanged.
  - :class:`LocalSentenceTransformersEmbeddings` loads a local
    sentence-transformers model. Heavy (~1.5GB resident with
    ``BAAI/bge-large-en-v1.5``); used as fallback when no remote
    endpoint is configured.

The factory :func:`get_embedder` reads :class:`app.config.Settings`:
``embedding_base_url`` chooses remote vs. local, ``embedding_model`` is
the model id, ``embedding_dim`` is the expected output dimension. The
expected dim is asserted at startup against the actual embedder so a
silently-misconfigured model doesn't write inconsistent BLOBs into
``world_facts``.

L2-normalisation happens at the boundary — output of :func:`embed` is
always unit-norm. AGENTS.md invariant #5: the cosine retrieval routine
in :mod:`app.llm.memory` reduces cosine similarity to a dot product on
the assumption that all stored vectors are normalised.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Protocol, runtime_checkable

import httpx
import numpy as np

from app.config import get_settings

log = logging.getLogger(__name__)


class EmbeddingError(RuntimeError):
    """Raised on embedding-backend failures (transport, dim mismatch,
    missing model). Unifies both backends' error surfaces so callers
    don't have to know which one's underneath."""


@runtime_checkable
class Embedder(Protocol):
    """The protocol every embedding backend implements."""

    @property
    def dim(self) -> int:
        """Output dimension. Stable for the lifetime of the embedder;
        validated at construction against ``settings.embedding_dim``."""
        ...

    async def embed(self, texts: list[str]) -> np.ndarray:
        """Return a ``(len(texts), dim)`` ``float32`` array, L2-normed
        per row. ``texts`` may be a single string in a list."""
        ...

    async def health(self) -> dict[str, Any]:
        """Liveness probe. Raises :class:`EmbeddingError` if the
        backend is non-functional. Cheap — call from app startup."""
        ...

    async def aclose(self) -> None:
        """Dispose underlying transport / model. Idempotent."""
        ...


def _l2_normalise(matrix: np.ndarray) -> np.ndarray:
    """Row-wise L2 normalisation. Zero-vectors stay zero (cosine with
    a zero vector is undefined; let the caller decide)."""

    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    safe = np.where(norms > 0, norms, 1.0)
    return (matrix / safe).astype(np.float32, copy=False)


# ---------------------------------------------------------------------------
# Remote: OpenAI-compatible /v1/embeddings
# ---------------------------------------------------------------------------


class OpenAIEmbeddings:
    """Async wrapper around an OpenAI-compatible ``/v1/embeddings``
    endpoint. The endpoint may be Ollama (``YOUR_AI_SERVER:11436``), a
    dedicated TEI server, or OpenAI itself — the request shape is
    identical."""

    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        expected_dim: int,
        timeout: httpx.Timeout | None = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._expected_dim = expected_dim
        self._timeout = timeout or httpx.Timeout(connect=5.0, read=30.0, write=10.0, pool=5.0)

    @property
    def dim(self) -> int:
        return self._expected_dim

    @property
    def model(self) -> str:
        return self._model

    async def embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self._expected_dim), dtype=np.float32)

        payload = {"model": self._model, "input": texts}
        try:
            async with httpx.AsyncClient(timeout=self._timeout) as c:
                response = await c.post(f"{self._base_url}/embeddings", json=payload)
                response.raise_for_status()
                body = response.json()
        except httpx.HTTPError as exc:
            raise EmbeddingError(f"embedding request failed: {exc}") from exc

        rows = body.get("data") or []
        if len(rows) != len(texts):
            raise EmbeddingError(f"embedding response had {len(rows)} rows, expected {len(texts)}")

        try:
            vectors = np.array([row["embedding"] for row in rows], dtype=np.float32)
        except (KeyError, ValueError, TypeError) as exc:
            raise EmbeddingError(f"malformed embedding response: {exc}") from exc

        if vectors.shape[1] != self._expected_dim:
            raise EmbeddingError(
                f"embedding dim mismatch: model returned {vectors.shape[1]},"
                f" settings.embedding_dim is {self._expected_dim}. Either swap"
                f" the model or update the setting."
            )
        return _l2_normalise(vectors)

    async def health(self) -> dict[str, Any]:
        # Quick canary: embed one short string. Confirms the model is
        # loaded on the server (Ollama gives a clear 404-with-model-name
        # if you haven't pulled the embedding model yet).
        vectors = await self.embed(["health-check"])
        return {
            "backend": "openai_compat",
            "base_url": self._base_url,
            "model": self._model,
            "dim": int(vectors.shape[1]),
            "norm_sample": float(np.linalg.norm(vectors[0])),
        }

    async def aclose(self) -> None:
        # httpx clients are constructed per-call; nothing to dispose.
        return None


# ---------------------------------------------------------------------------
# Local: sentence-transformers
# ---------------------------------------------------------------------------


class LocalSentenceTransformersEmbeddings:
    """Local fallback. The model loads on first :func:`embed` call (so
    construction is cheap) but blocks the loop briefly while torch
    spins up. Call :func:`health` at app startup to pay that cost on
    boot rather than on the first player turn."""

    def __init__(self, *, model: str, expected_dim: int) -> None:
        self._model_name = model
        self._expected_dim = expected_dim
        self._model: Any | None = None
        self._load_lock = asyncio.Lock()

    @property
    def dim(self) -> int:
        return self._expected_dim

    @property
    def model(self) -> str:
        return self._model_name

    async def _ensure_loaded(self) -> Any:
        cached = self._model
        if cached is not None:
            return cached
        async with self._load_lock:
            cached = self._model
            if cached is not None:
                return cached
            log.info("loading local sentence-transformers model: %s", self._model_name)
            # The import is local to keep ``app.llm.embeddings`` importable
            # in environments where torch isn't installed (e.g. a deployment
            # that uses only the remote backend).
            from sentence_transformers import SentenceTransformer

            model = await asyncio.to_thread(SentenceTransformer, self._model_name)
            reported_dim = model.get_sentence_embedding_dimension()
            if reported_dim is None:
                raise EmbeddingError(f"local model {self._model_name!r} did not report a dimension")
            actual_dim = int(reported_dim)
            if actual_dim != self._expected_dim:
                raise EmbeddingError(
                    f"local model {self._model_name!r} produced dim={actual_dim},"
                    f" but settings.embedding_dim is {self._expected_dim}."
                )
            self._model = model
            return model

    async def embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self._expected_dim), dtype=np.float32)
        model = await self._ensure_loaded()
        # ``encode`` is CPU/GPU-bound; offload to a thread so we don't
        # block the event loop. ``normalize_embeddings=True`` does its
        # own L2 norm — we re-normalise just to be sure (sentence-transformers
        # has historically been sloppy about edge cases like all-zero
        # outputs from very short strings).
        raw = await asyncio.to_thread(
            model.encode,
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        vectors = np.asarray(raw, dtype=np.float32)
        if vectors.shape[1] != self._expected_dim:
            raise EmbeddingError(
                f"local model dim mismatch at runtime: {vectors.shape[1]}"
                f" vs expected {self._expected_dim}"
            )
        return _l2_normalise(vectors)

    async def health(self) -> dict[str, Any]:
        await self._ensure_loaded()
        vectors = await self.embed(["health-check"])
        return {
            "backend": "sentence_transformers",
            "model": self._model_name,
            "dim": int(vectors.shape[1]),
            "norm_sample": float(np.linalg.norm(vectors[0])),
        }

    async def aclose(self) -> None:
        # Drop the model reference; the GC reclaims its memory once
        # nothing else holds it. Idempotent.
        self._model = None


# ---------------------------------------------------------------------------
# Factory + singleton
# ---------------------------------------------------------------------------

_singleton: Embedder | None = None


def _build_embedder() -> Embedder:
    """Pick the right backend based on settings."""

    settings = get_settings()
    if settings.embedding_base_url:
        log.info(
            "embeddings: remote backend at %s (model=%s, dim=%d)",
            settings.embedding_base_url,
            settings.embedding_model,
            settings.embedding_dim,
        )
        return OpenAIEmbeddings(
            base_url=settings.embedding_base_url,
            model=settings.embedding_model,
            expected_dim=settings.embedding_dim,
        )
    log.info(
        "embeddings: local sentence-transformers backend (model=%s, dim=%d)",
        settings.embedding_model,
        settings.embedding_dim,
    )
    return LocalSentenceTransformersEmbeddings(
        model=settings.embedding_model,
        expected_dim=settings.embedding_dim,
    )


def get_embedder() -> Embedder:
    """Process-wide :class:`Embedder` singleton. Built on first call."""

    global _singleton
    if _singleton is None:
        _singleton = _build_embedder()
    return _singleton


async def reset_for_tests() -> None:
    """Drop the singleton so a test can swap settings and rebuild."""

    global _singleton
    if _singleton is not None:
        await _singleton.aclose()
        _singleton = None


__all__ = [
    "Embedder",
    "EmbeddingError",
    "LocalSentenceTransformersEmbeddings",
    "OpenAIEmbeddings",
    "get_embedder",
    "reset_for_tests",
]
