"""Phase 6.6 tests for the dashboard's session affordances.

A real-play playthrough surfaced that the featured campaign's
Start Session affordance was a ``<form>`` whose visible elements
were all ``<span>``-wrapped — no submit button, so clicking the
card did nothing. This file covers both the empty-session and
active-session paths so the regression class ("design mock always
had a session, empty path never exercised") doesn't recur.
"""

from __future__ import annotations

import re

import pytest
from httpx import AsyncClient

_VALID_PW = "correct horse battery staple"


async def _register_and_login(client: AsyncClient, username: str = "alice") -> str:
    response = await client.post(
        "/api/auth/register",
        json={"username": username, "password": _VALID_PW},
    )
    assert response.status_code == 201
    return response.json()["id"]  # type: ignore[no-any-return]


async def _create_campaign(client: AsyncClient, name: str = "Test Camp") -> str:
    response = await client.post("/api/campaigns", json={"name": name})
    assert response.status_code == 201
    return response.json()["id"]  # type: ignore[no-any-return]


# ---------------------------------------------------------------------------
# Featured campaign: empty session path
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_dashboard_empty_session_renders_interactive_start(
    client: AsyncClient,
) -> None:
    """A campaign with no active session must render Start Session
    as an interactive element — a real ``<button type="submit">``
    inside a ``<form>`` pointed at the create-session endpoint, not
    a span the user can't click.
    """

    await _register_and_login(client)
    campaign_id = await _create_campaign(client, "Empty Camp")

    response = await client.get("/dashboard")
    assert response.status_code == 200
    body = response.text

    # The featured campaign has no active session, so the dashboard
    # must render a form posting to .../sessions and contain a real
    # submit button. The pre-Phase-6.6 bug was a span-only arrow.
    expected_action = f'action="/api/campaigns/{campaign_id}/sessions"'
    assert expected_action in body, "Start Session form must point at /sessions"

    # A submit button must exist somewhere inside the dashboard markup.
    # Match on the literal action so we know it belongs to the right form.
    form_match = re.search(
        r'<form[^>]*action="/api/campaigns/'
        + re.escape(campaign_id)
        + r'/sessions"[^>]*>(.*?)</form>',
        body,
        flags=re.DOTALL,
    )
    assert form_match is not None, "Start Session <form> not found"
    form_inner = form_match.group(1)
    assert 'type="submit"' in form_inner, (
        'Start Session form must contain a <button type="submit"> — '
        "without one the form never emits a submit event"
    )

    # And the diegetic copy is still on the affordance. Note: the
    # featured-campaign block uses "Start Session" with a capital S;
    # the small-row variant uses "Start session". Either form is a
    # valid signal that the affordance is wired.
    assert "Start Session" in body or "Start session" in body


# ---------------------------------------------------------------------------
# Featured campaign: active session path
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_dashboard_active_session_renders_resume_link(
    client: AsyncClient,
) -> None:
    """When there is an active session the affordance is a plain
    ``<a href="/play/{id}">`` — no form, no JS required. Resume
    Session has been working since Phase 6; this guards against a
    refactor that breaks it."""

    await _register_and_login(client)
    campaign_id = await _create_campaign(client, "Resume Camp")
    session = (await client.post(f"/api/campaigns/{campaign_id}/sessions")).json()
    session_id = session["id"]

    response = await client.get("/dashboard")
    assert response.status_code == 200
    body = response.text

    # Resume Session is rendered as an anchor; no submit button needed.
    assert f'href="/play/{session_id}"' in body
    assert "Resume Session" in body


# ---------------------------------------------------------------------------
# End-to-end: Start Session creates a session and the user can navigate
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_dashboard_start_session_creates_and_redirects(
    client: AsyncClient,
) -> None:
    """Mirror the click path: render the dashboard, find the Start
    Session form's action, POST to it, follow the implied redirect
    to /play/{id} and assert the play screen loads. Catches any
    regression that decouples the form action from the create-
    session endpoint."""

    await _register_and_login(client)
    campaign_id = await _create_campaign(client, "Click Camp")

    page = await client.get("/dashboard")
    assert page.status_code == 200
    # The dashboard JS reads the form's action; mirror that here.
    expected_action = f"/api/campaigns/{campaign_id}/sessions"
    assert f'action="{expected_action}"' in page.text

    # POST to that action — same call the JS makes on submit.
    create = await client.post(expected_action)
    assert create.status_code == 201
    session_id = create.json()["id"]

    # Follow the implied redirect-to-session.
    play = await client.get(f"/play/{session_id}")
    assert play.status_code == 200


# ---------------------------------------------------------------------------
# Other campaigns row: empty session path
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_dashboard_other_campaigns_start_is_interactive(
    client: AsyncClient,
) -> None:
    """The same fix applies to the smaller campaign-row variant —
    Start session in the other-campaigns list is also a full-card
    button so clicking the body works, not just the tiny arrow."""

    await _register_and_login(client)
    # Featured slot goes to the most-recently-played campaign, which
    # we leave with an active session. The "other" campaign without
    # a session is the one we want to inspect.
    featured_id = await _create_campaign(client, "Featured")
    await client.post(f"/api/campaigns/{featured_id}/sessions")
    other_id = await _create_campaign(client, "Other Empty")

    response = await client.get("/dashboard")
    assert response.status_code == 200
    body = response.text

    # The empty-session "other campaign" form must contain a submit.
    other_form = re.search(
        r'<form[^>]*action="/api/campaigns/'
        + re.escape(other_id)
        + r'/sessions"[^>]*>(.*?)</form>',
        body,
        flags=re.DOTALL,
    )
    assert other_form is not None, "small Start session <form> not found"
    assert 'type="submit"' in other_form.group(1)
