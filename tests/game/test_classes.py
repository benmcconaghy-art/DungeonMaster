"""Tests for ``app.game.classes`` (loader + cache)."""

from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import pytest
from pydantic import ValidationError

from app.game import classes


def test_loader_validates_and_indexes_by_name(classes_yaml: Path) -> None:
    table = classes.load_classes(classes_yaml)
    assert set(table) == {"Fighter", "Thief", "Cleric"}
    fighter = table["Fighter"]
    assert fighter.hit_die == "d8"
    assert fighter.saves[1].death_ray == 12
    assert fighter.attack_bonus_progression[1] == 1


def test_get_class_by_name(classes_yaml: Path) -> None:
    classes.load_classes(classes_yaml)
    fighter = classes.get_class("Fighter")
    assert fighter.name == "Fighter"


def test_unknown_class_raises(classes_yaml: Path) -> None:
    classes.load_classes(classes_yaml)
    with pytest.raises(KeyError, match="unknown class"):
        classes.get_class("Wizard")


def test_thief_skills_table_loaded(classes_yaml: Path) -> None:
    table = classes.load_classes(classes_yaml)
    thief = table["Thief"]
    assert thief.thief_skills is not None
    assert thief.thief_skills[1].open_locks == 25


def test_xp_progression_null_passes_validator(tmp_path: Path) -> None:
    """``xp_progression: null`` should validate (pre-validator returns value as-is)."""

    p = tmp_path / "classes.yaml"
    p.write_text(
        dedent(
            """\
            - name: Bard
              hit_die: d6
              prime_requisite: CHA
              weapon_restrictions: any
              armour_restrictions: leather only
              saves:
                1: {death_ray: 13, magic_wand: 14, paralysis: 13, dragon_breath: 16, spells: 15}
              attack_bonus_progression: {1: 1}
              xp_progression: null
            """
        ),
        encoding="utf-8",
    )
    table = classes.load_classes(p)
    assert table["Bard"].xp_progression is None


def test_top_level_must_be_list(tmp_path: Path) -> None:
    bad = tmp_path / "classes.yaml"
    bad.write_text("name: not a list", encoding="utf-8")
    with pytest.raises(ValueError, match="expected a top-level list"):
        classes.load_classes(bad)


def test_extra_field_rejected_at_validation(tmp_path: Path) -> None:
    bad = tmp_path / "classes.yaml"
    bad.write_text(
        dedent(
            """\
            - name: Bandit
              hit_die: d6
              prime_requisite: STR
              weapon_restrictions: any
              armour_restrictions: any
              saves:
                1: {death_ray: 12, magic_wand: 13, paralysis: 14, dragon_breath: 15, spells: 17}
              attack_bonus_progression: {1: 1}
              completely_made_up_field: 42
            """
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValidationError):
        classes.load_classes(bad)


def test_cache_reuses_first_call(classes_yaml: Path, tmp_path: Path) -> None:
    classes.load_classes(classes_yaml)
    other = tmp_path / "other.yaml"
    other.write_text(
        dedent(
            """\
            - name: Magic-User
              hit_die: d4
              prime_requisite: INT
              weapon_restrictions: dagger only
              armour_restrictions: none
              saves:
                1: {death_ray: 13, magic_wand: 14, paralysis: 13, dragon_breath: 16, spells: 15}
              attack_bonus_progression: {1: 1}
            """
        ),
        encoding="utf-8",
    )
    # Second call passes a different path but cache wins (per loader contract).
    table = classes.load_classes(other)
    assert "Magic-User" not in table
    assert "Fighter" in table


def test_reset_cache_clears_state(classes_yaml: Path, tmp_path: Path) -> None:
    classes.load_classes(classes_yaml)
    classes.reset_cache()
    other = tmp_path / "other.yaml"
    other.write_text(
        dedent(
            """\
            - name: Cleric
              hit_die: d6
              prime_requisite: WIS
              weapon_restrictions: blunt only
              armour_restrictions: any
              saves:
                1: {death_ray: 11, magic_wand: 12, paralysis: 14, dragon_breath: 16, spells: 15}
              attack_bonus_progression: {1: 1}
            """
        ),
        encoding="utf-8",
    )
    table = classes.load_classes(other)
    assert "Cleric" in table
