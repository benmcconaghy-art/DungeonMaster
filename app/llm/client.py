"""vLLM client wrapper.

Thin async wrapper around the OpenAI SDK pointed at our internal vLLM
endpoint serving Nemotron 3 Super. Exposes ``stream_dm`` and the
embeddings call used by the world-fact retriever.

Phase 2 implements this; Phase 0 ships only the placeholder.
"""

from __future__ import annotations
