"""Tests for ``app.images.portrait`` — prompt composition + queue
enqueue helpers used by the canonical portrait flow (Step 7).

Real Valkey is not exercised here; an in-memory fake stands in for
the redis client. Round-trip with a real Valkey is covered by
``tests/images/test_queue.py``.
"""

from __future__ import annotations

import json
from typing import Any, cast

import pytest
import redis.asyncio as redis_async

from app.images import portrait as portrait_module
from app.images.portrait import (
    build_portrait_prompt,
    enqueue_portrait,
    enqueue_scene,
    get_queue_client,
    reset_for_tests,
    set_queue_client_for_tests,
)
from app.images.queue import QUEUE_KEY, ImageJob

# ---------------------------------------------------------------------------
# build_portrait_prompt
# ---------------------------------------------------------------------------


def test_build_prompt_with_full_pc_fields() -> None:
    """A character rolled via chargen has race + class + alignment.
    The prompt should embed all of them in a stable order so the
    dedup hash is deterministic across calls."""

    prompt = build_portrait_prompt(
        name="Brunhild",
        race="Human",
        class_name="Fighter",
        alignment="Lawful",
        description="Tall, copper-braided, scarred jaw.",
    )
    assert "Brunhild" in prompt
    assert "Human" in prompt
    assert "Fighter" in prompt
    assert "Lawful" in prompt
    assert "copper-braided" in prompt


def test_build_prompt_npc_minimal_fields() -> None:
    """An NPC the LLM spawned via ``spawn_npc`` may have only a name
    and description — no race/class/alignment. The prompt must still
    be coherent (no "a  ," gaps, no trailing punctuation)."""

    prompt = build_portrait_prompt(
        name="Castellan Thorvald",
        description="Greying veteran, missing two fingers on his left hand.",
    )
    assert prompt.startswith("Portrait of Castellan Thorvald")
    assert "Greying veteran" in prompt
    # No empty descriptor placeholder.
    assert ", a , " not in prompt
    assert ", a a " not in prompt


def test_build_prompt_strips_trailing_period_from_description() -> None:
    """Avoid double-period after appending the campaign style suffix.
    If description is "X." and the worker appends "Style: Y", a
    trailing period would produce "X..  Style: Y"."""

    prompt = build_portrait_prompt(name="X", description="A weathered face.")
    assert "face." not in prompt[-10:]
    assert prompt.endswith("A weathered face")


def test_build_prompt_is_deterministic() -> None:
    """Same inputs → same output. The dedup hash on the worker side
    will only short-circuit if the prompt builder is stable."""

    a = build_portrait_prompt(name="Tav", race="Elf", class_name="Wizard")
    b = build_portrait_prompt(name="Tav", race="Elf", class_name="Wizard")
    assert a == b


def test_build_prompt_handles_only_class_no_race() -> None:
    """Some NPC stat blocks have a class but no race. Make sure the
    'a Fighter' phrasing reads cleanly without an awkward 'a   Fighter'."""

    prompt = build_portrait_prompt(name="Ned", class_name="Fighter")
    assert "a Fighter" in prompt


# ---------------------------------------------------------------------------
# enqueue_portrait
# ---------------------------------------------------------------------------


class _FakeRedis:
    """Captures ``rpush`` calls in a list so tests can assert the
    queue payload without needing a real Valkey."""

    def __init__(self) -> None:
        self.pushed: list[tuple[str, bytes]] = []

    async def rpush(self, key: str, value: bytes) -> int:
        self.pushed.append((key, value))
        return len(self.pushed)

    async def aclose(self) -> None:
        return None


@pytest.mark.asyncio
async def test_enqueue_portrait_pushes_npc_kind_job() -> None:
    """Portraits use kind='npc' (768x1024/32-step per spec §8) — not
    'scene'. A regression that sent kind='scene' would silently
    produce wide landscape portraits."""

    fake = _FakeRedis()
    image_id = await enqueue_portrait(
        cast(redis_async.Redis, fake),
        campaign_id="camp-1",
        prompt="Portrait of Tav",
        subject_character_id="char-1",
    )

    assert len(fake.pushed) == 1
    key, raw = fake.pushed[0]
    assert key == QUEUE_KEY

    job = ImageJob.model_validate(json.loads(raw))
    assert job.id == image_id
    assert job.campaign_id == "camp-1"
    assert job.kind == "npc"
    assert job.prompt == "Portrait of Tav"
    assert job.subject_character_id == "char-1"
    assert job.subject_npc_id is None


@pytest.mark.asyncio
async def test_enqueue_portrait_rejects_both_subject_ids() -> None:
    """One image cannot be canonical for both a PC and an NPC. The
    helper rejects this misuse before pushing — otherwise the worker
    would link the same image to both ``characters.canonical_image_id``
    and ``npcs.canonical_image_id``, which is nonsensical."""

    fake = _FakeRedis()
    with pytest.raises(ValueError, match="only one of"):
        await enqueue_portrait(
            cast(redis_async.Redis, fake),
            campaign_id="c",
            prompt="x",
            subject_character_id="char-1",
            subject_npc_id="npc-1",
        )
    # Nothing pushed — the validation fires before rpush.
    assert fake.pushed == []


@pytest.mark.asyncio
async def test_enqueue_portrait_session_id_round_trips() -> None:
    """The session id, when set, threads through onto the job so the
    worker knows where to broadcast the eventual ``image_ready``."""

    fake = _FakeRedis()
    await enqueue_portrait(
        cast(redis_async.Redis, fake),
        campaign_id="c",
        prompt="x",
        session_id="sess-9",
        subject_npc_id="npc-1",
    )
    job = ImageJob.model_validate(json.loads(fake.pushed[0][1]))
    assert job.session_id == "sess-9"
    assert job.subject_npc_id == "npc-1"


@pytest.mark.asyncio
async def test_enqueue_portrait_no_subject_is_allowed() -> None:
    """A portrait with no subject FK persists but doesn't link to any
    character or NPC. Useful for one-off requests where the caller
    just wants the image — neither subject id set is fine."""

    fake = _FakeRedis()
    image_id = await enqueue_portrait(
        cast(redis_async.Redis, fake),
        campaign_id="c",
        prompt="a generic adventurer",
    )
    assert image_id
    job = ImageJob.model_validate(json.loads(fake.pushed[0][1]))
    assert job.subject_character_id is None
    assert job.subject_npc_id is None


# ---------------------------------------------------------------------------
# enqueue_scene
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_enqueue_scene_plain_generate_job() -> None:
    """No reference set → plain /generate. Job carries the kind the
    caller picked (default 'scene') and no edit fields."""

    fake = _FakeRedis()
    image_id = await enqueue_scene(
        cast(redis_async.Redis, fake),
        campaign_id="c",
        prompt="a goblin warband on a forest road",
        kind="scene",
        session_id="sess-1",
    )
    job = ImageJob.model_validate(json.loads(fake.pushed[0][1]))
    assert job.id == image_id
    assert job.kind == "scene"
    assert job.session_id == "sess-1"
    assert job.reference_image_id is None
    assert job.edit_instruction is None


@pytest.mark.asyncio
async def test_enqueue_scene_edit_job_carries_reference_and_instruction() -> None:
    """Reference + instruction set → Kontext /edit. The worker
    dispatches via flux.edit when reference_image_id is non-null;
    the edit_instruction becomes the prompt argument to /edit."""

    fake = _FakeRedis()
    image_id = await enqueue_scene(
        cast(redis_async.Redis, fake),
        campaign_id="c",
        prompt="same character, kneeling beside a fallen companion",
        reference_image_id="canon-1",
        edit_instruction="same character, kneeling beside a fallen companion",
    )
    job = ImageJob.model_validate(json.loads(fake.pushed[0][1]))
    assert job.id == image_id
    assert job.reference_image_id == "canon-1"
    assert job.edit_instruction == "same character, kneeling beside a fallen companion"


@pytest.mark.asyncio
async def test_enqueue_scene_reference_without_instruction_rejected() -> None:
    """Reference id without an edit instruction is malformed — the
    worker would emit ``invalid_job``. Catch it at the enqueuer to
    fail fast with a useful Python traceback."""

    fake = _FakeRedis()
    with pytest.raises(ValueError, match="edit_instruction"):
        await enqueue_scene(
            cast(redis_async.Redis, fake),
            campaign_id="c",
            prompt="x",
            reference_image_id="canon-1",
        )
    assert fake.pushed == []


@pytest.mark.asyncio
async def test_enqueue_scene_instruction_without_reference_rejected() -> None:
    """Symmetric: an edit instruction with no reference image makes
    no sense — there's nothing to edit."""

    fake = _FakeRedis()
    with pytest.raises(ValueError, match="reference_image_id"):
        await enqueue_scene(
            cast(redis_async.Redis, fake),
            campaign_id="c",
            prompt="x",
            edit_instruction="say something",
        )
    assert fake.pushed == []


# ---------------------------------------------------------------------------
# Singleton wiring
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_set_queue_client_for_tests_overrides_singleton() -> None:
    """Tests need to inject a fake redis client without monkeypatching
    every call site. The setter swaps the module singleton in-place."""

    await reset_for_tests()
    fake = _FakeRedis()
    set_queue_client_for_tests(cast(Any, fake))
    try:
        assert cast(Any, get_queue_client()) is fake
    finally:
        await reset_for_tests()
        assert portrait_module._singleton is None
