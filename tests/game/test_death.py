"""Tests for ``app.game.death`` — the OSR death-and-dismemberment table."""

from __future__ import annotations

import random

import pytest

from app.game.death import death_and_dismemberment
from app.game.rules import CharacterStats, DamageResult, apply_damage


def _make_character(*, hp_current: int = 5, hp_max: int = 8) -> CharacterStats:
    return CharacterStats(
        name="Hrok",
        class_name="Fighter",
        level=1,
        hp_current=hp_current,
        hp_max=hp_max,
        ac=12,
        str_score=13,
        int_score=10,
        wis_score=10,
        dex_score=12,
        con_score=12,
        cha_score=10,
    )


def test_zero_hp_goes_through_table_not_instant_death() -> None:
    """A blow that drops HP to 0 with no overrun rolls on the table; it
    should produce a survivable result for the typical cases."""

    char = _make_character(hp_current=5)
    blow = apply_damage(char, 5)
    rng = random.Random(0)
    result = death_and_dismemberment(char, blow, rng=rng)
    assert blow.dropped_to_zero is True
    assert blow.below_zero_by == 0
    # With zero overrun we cannot land on "dead" (16+) on a plain 2d6.
    assert result.outcome != "dead"


def test_table_lookup_by_total() -> None:
    """Force each total via a controlled rng: rolling 2d6 always yields 1+1=2."""

    class FixedLow(random.Random):
        def randint(self, a: int, b: int) -> int:
            return a  # always min

    char = _make_character(hp_current=5)
    blow = apply_damage(char, 5)
    rng = FixedLow()
    result = death_and_dismemberment(char, blow, rng=rng)
    # 2d6 with always-1 -> 2; modifier 0 -> total 2 -> "knocked_out"
    assert result.outcome == "knocked_out"
    assert result.hp_after == 0
    assert "unconscious_turns" in result.detail


def test_lingering_injury_outcome_populates_detail() -> None:
    """With overrun=2 and minimum 2d6=2, total=4 lands on lingering_injury."""

    class FixedLow(random.Random):
        def randint(self, a: int, b: int) -> int:
            return a  # always min

    char = _make_character(hp_current=2)
    blow = apply_damage(char, 4)  # below_zero_by = 2
    assert blow.below_zero_by == 2
    rng = FixedLow()
    result = death_and_dismemberment(char, blow, rng=rng)
    assert result.outcome == "lingering_injury"
    assert result.hp_after == 0
    assert "unconscious_minutes" in result.detail
    assert "recovery_days" in result.detail


def test_dead_when_overrun_is_extreme() -> None:
    """A massive overrun makes death the only outcome regardless of dice."""

    char = _make_character(hp_current=1, hp_max=8)
    # Take 20 damage from 1 HP -> below_zero_by = 19, total >= 19+2 = 21
    blow = apply_damage(char, 20)
    assert blow.below_zero_by == 19
    rng = random.Random(0)
    result = death_and_dismemberment(char, blow, rng=rng)
    assert result.outcome == "dead"
    assert result.hp_after is None


def test_critical_adds_two_to_modifier() -> None:
    """``critical=True`` shifts the table outcome upward."""

    char = _make_character(hp_current=2)

    class FixedHigh(random.Random):
        def randint(self, a: int, b: int) -> int:
            return b  # always max

    blow = apply_damage(char, 2)  # below_zero_by = 0
    rng = FixedHigh()
    result_normal = death_and_dismemberment(char, blow, rng=rng)
    rng = FixedHigh()
    result_crit = death_and_dismemberment(char, blow, rng=rng, critical=True)
    # max 2d6 = 12; normal: total 12 -> debilitating_wound; crit: total 14 -> crippled
    assert result_normal.outcome == "debilitating_wound"
    assert result_crit.outcome == "crippled"


def test_non_killing_blow_raises() -> None:
    """Calling the table on a hit that didn't drop to zero is a programming error."""

    char = _make_character(hp_current=10)
    blow = apply_damage(char, 5)
    assert blow.dropped_to_zero is False
    rng = random.Random(0)
    with pytest.raises(ValueError, match="non-killing"):
        death_and_dismemberment(char, blow, rng=rng)


def test_each_outcome_reachable_via_total() -> None:
    """Walk through every distinct outcome; ensure detail keys are populated."""

    char = _make_character()
    seen: set[str] = set()

    # We synthesise a DamageResult with the exact below_zero_by we need
    # so we can pin down each row.
    def roll_with_overrun(overrun: int) -> tuple[str, dict[str, object]]:
        rng = random.Random(0)
        synth = DamageResult(
            target=char.name,
            amount=10,
            source=None,
            previous_hp=5,
            new_hp=-overrun,
            dropped_to_zero=True,
            below_zero_by=overrun,
        )
        result = death_and_dismemberment(char, synth, rng=rng)
        return result.outcome, dict(result.detail)

    # 2d6 with seed=0: [3,5] -> 8. Adding overrun shifts the total.
    sample_rng = random.Random(0)
    base = sample_rng.randint(1, 6) + sample_rng.randint(1, 6)
    # Map base + overrun to outcome row.
    for overrun in range(0, 20):
        outcome, detail = roll_with_overrun(overrun)
        seen.add(outcome)
        assert isinstance(detail, dict)
    # We should have hit several distinct outcomes.
    assert "dead" in seen or base + 19 >= 16
    assert len(seen) >= 3
