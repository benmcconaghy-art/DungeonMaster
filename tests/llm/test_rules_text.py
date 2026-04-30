"""Tests for ``app.llm.rules_text`` — the condensed BFRPG rules block."""

from __future__ import annotations

from app.llm import rules_text


def test_render_rules_text_mentions_core_mechanics() -> None:
    """Output names ascending AC, the five save categories, and BFRPG core."""

    rules_text.reset_cache()
    text = rules_text.render_rules_text()

    assert "Ascending AC only" in text
    # Save categories from the BFRPG five-bucket model.
    for cat in ("Death Ray", "Magic Wand", "Paralysis", "Dragon Breath"):
        assert cat in text
    # House-rule discipline cues for the LLM.
    assert "request_dice_roll" in text
    assert "apply_damage" in text


def test_rules_text_includes_class_archetypes() -> None:
    """Each canonical class is named at least once."""

    rules_text.reset_cache()
    text = rules_text.render_rules_text()
    for cls in ("Fighter", "Cleric", "Magic-User", "Thief"):
        assert cls in text


def test_rules_text_mentions_ability_modifier_curve() -> None:
    """The 3..18 → modifier table is rendered. Spot-check the endpoints."""

    rules_text.reset_cache()
    text = rules_text.render_rules_text()
    # Sentinel values from app.game.rules.ability_modifier
    assert "3=-3" in text
    assert "18=+3" in text


def test_rules_text_caches_between_calls() -> None:
    """Repeated renders return the same string object (cache hit)."""

    rules_text.reset_cache()
    first = rules_text.render_rules_text()
    second = rules_text.render_rules_text()
    assert first is second


def test_rules_text_resets_after_cache_clear() -> None:
    """``reset_cache`` lets the next call re-render fresh."""

    rules_text.reset_cache()
    first = rules_text.render_rules_text()
    rules_text.reset_cache()
    second = rules_text.render_rules_text()
    # New string object after reset (content equal, identity not).
    assert first == second
    assert first is not second
