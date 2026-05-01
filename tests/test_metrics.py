"""Phase 7 /metrics endpoint tests.

* ``GET /metrics`` returns 200 with a Prometheus exposition body.
* The exposition includes the metric names we declared in
  ``app.metrics`` so an operator pointing Prometheus at the endpoint
  doesn't have to guess.
* Hitting an HTTP endpoint bumps the ``dm_http_requests_total``
  counter — proves the AccessLogMiddleware instrumentation is wired
  end-to-end.
* An LLM ``complete`` call bumps ``dm_llm_calls_total`` — proves the
  llm-client instrumentation site is wired.
"""

from __future__ import annotations

import pytest
from httpx import AsyncClient
from prometheus_client.parser import text_string_to_metric_families


@pytest.mark.asyncio
async def test_metrics_endpoint_returns_prometheus_exposition(
    client: AsyncClient,
) -> None:
    """Endpoint returns 200 with a body parseable by the Prometheus
    text-format parser. Validates the contract a real Prometheus
    server would assert against."""

    r = await client.get("/metrics")
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/plain")

    # Parse the body — text_string_to_metric_families raises on
    # malformed exposition. The parser strips the ``_total`` suffix
    # from counter family names; the rendered samples carry it.
    families = list(text_string_to_metric_families(r.text))
    family_names = {fam.name for fam in families}
    assert "dm_http_requests" in family_names  # rendered as dm_http_requests_total
    assert "dm_llm_calls" in family_names
    assert "dm_tool_dispatch" in family_names
    assert "dm_image_jobs" in family_names
    # Confirm the rendered samples carry the canonical _total suffix.
    assert "dm_http_requests_total" in r.text
    assert "dm_llm_calls_total" in r.text


@pytest.mark.asyncio
async def test_http_request_counter_increments(client: AsyncClient) -> None:
    """A request to a real endpoint increments the http_requests_total
    counter for that path template + method + status combination."""

    # Prime: hit /health a couple times. /health uses the literal
    # path (no path params), so the path label is /health.
    for _ in range(3):
        r = await client.get("/health")
        assert r.status_code == 200

    metrics_resp = await client.get("/metrics")
    body = metrics_resp.text
    # The Prometheus exposition emits a sample like:
    #   dm_http_requests_total{method="GET",path="/health",status="200"} 3.0
    # Sample-line matching keeps the assertion robust to label order.
    health_lines = [
        line
        for line in body.splitlines()
        if line.startswith("dm_http_requests_total")
        and 'path="/health"' in line
        and 'method="GET"' in line
        and 'status="200"' in line
    ]
    assert len(health_lines) == 1
    # Last whitespace-separated field is the value.
    value = float(health_lines[0].rsplit(" ", 1)[1])
    assert value >= 3.0


@pytest.mark.asyncio
async def test_llm_call_counter_increments(client: AsyncClient) -> None:
    """A successful complete() call bumps dm_llm_calls_total. We mock
    the openai client at the boundary to keep the test offline."""

    from app import metrics
    from app.llm.client import DmClient

    # Build a fresh DmClient instance (don't go through the singleton
    # — that depends on the running Settings + base URL). Stub out the
    # underlying openai client so .chat.completions.create returns a
    # canned response with a usage block.
    dm = DmClient.__new__(DmClient)
    dm._resolved_model = "test-model"  # type: ignore[attr-defined]

    class _Choice:
        message = type("M", (), {"content": "ok"})()

    class _Usage:
        prompt_tokens = 10
        completion_tokens = 5
        total_tokens = 15

    class _Resp:
        # The test stub presents an openai-response duck type; the
        # test creates one instance per call, so the "mutable class
        # attribute" warning is irrelevant here.
        choices = [_Choice()]  # noqa: RUF012
        usage = _Usage()

    class _OpenAIStub:
        class chat:
            class completions:
                @staticmethod
                async def create(**kwargs: object) -> object:
                    return _Resp()

    dm._client = _OpenAIStub()  # type: ignore[attr-defined]

    # Snapshot counter value before; call complete; expect +1 for the
    # ok bucket and the right token totals.
    before = metrics.llm_calls_total.labels(
        model="test-model", reasoning_mode="full", outcome="ok"
    )._value.get()
    await dm.complete([{"role": "user", "content": "hi"}])
    after = metrics.llm_calls_total.labels(
        model="test-model", reasoning_mode="full", outcome="ok"
    )._value.get()
    assert after == before + 1

    prompt_count = metrics.llm_tokens_total.labels(model="test-model", kind="prompt")._value.get()
    assert prompt_count >= 10
    completion_count = metrics.llm_tokens_total.labels(
        model="test-model", kind="completion"
    )._value.get()
    assert completion_count >= 5


@pytest.mark.asyncio
async def test_path_label_uses_route_template_not_resolved_url(
    client: AsyncClient,
) -> None:
    """Hitting /api/campaigns/<some-uuid> records the metric under the
    template ``/api/campaigns/{campaign_id}``, not the resolved URL.
    Without this, label cardinality grows linearly with campaign count."""

    # Need to be authenticated to hit campaign detail; create an account.
    await client.post(
        "/api/auth/register",
        json={"username": "metrics_user", "password": "correct horse battery"},
    )
    create = await client.post("/api/campaigns", json={"name": "Metric Test"})
    assert create.status_code == 201
    cid = create.json()["id"]

    # Two distinct path-param values; should bin to one label set.
    detail = await client.get(f"/api/campaigns/{cid}")
    assert detail.status_code == 200

    # Now confirm the metric uses the template.
    body = (await client.get("/metrics")).text
    # The route template in the FastAPI router is
    # /api/campaigns/{campaign_id}; we check for either that or the
    # resolved literal — fail if the literal slipped in.
    assert f'path="/api/campaigns/{cid}"' not in body
    assert 'path="/api/campaigns/{campaign_id}"' in body
