"""Tests for ``app.game.chargen``."""

from __future__ import annotations

import random
from pathlib import Path

import pytest

from app.game import classes, races
from app.game.chargen import (
    AbilityScores,
    generate_character,
    roll_abilities,
    starting_ac,
    starting_gold,
    starting_hp,
)


def test_starting_hp_minimum_one_with_negative_con() -> None:
    # d4 with -3 con: 1 - 3 = -2, clamped to 1
    assert starting_hp("d4", -3) == 1


def test_starting_hp_max_die_plus_con() -> None:
    assert starting_hp("d8", 0) == 8
    assert starting_hp("d8", 2) == 10
    assert starting_hp("d4", 1) == 5


def test_starting_ac_default() -> None:
    assert starting_ac(0) == 11
    assert starting_ac(2) == 13
    assert starting_ac(-1) == 10


def test_starting_gold_in_bfrpg_range() -> None:
    rng = random.Random(0)
    g = starting_gold(rng=rng)
    # 3d6 = 3..18; *10 -> 30..180
    assert 30 <= g <= 180
    assert g % 10 == 0


def test_roll_abilities_classic_in_3_18_range() -> None:
    rng = random.Random(0)
    scores = roll_abilities("classic", rng=rng)
    for ability in ("str", "int", "wis", "dex", "con", "cha"):
        assert 3 <= scores.get(ability) <= 18


def test_roll_abilities_heroic_4d6kh3() -> None:
    rng = random.Random(0)
    scores = roll_abilities("heroic", rng=rng)
    # 4d6 dropping the lowest is in range 3..18 too, but expected mean is higher
    for ability in ("str", "int", "wis", "dex", "con", "cha"):
        assert 3 <= scores.get(ability) <= 18


def test_generate_character_full_pipeline(classes_yaml: Path, races_yaml: Path) -> None:
    classes.load_classes(classes_yaml)
    races.load_races(races_yaml)
    abilities = AbilityScores(
        str_score=14,
        int_score=10,
        wis_score=10,
        dex_score=12,
        con_score=14,
        cha_score=10,
    )
    rng = random.Random(0)
    char = generate_character(
        name="Hrok",
        race_name="Human",
        class_name="Fighter",
        alignment="lawful",
        rng=rng,
        abilities=abilities,
    )
    assert char.name == "Hrok"
    assert char.class_name == "Fighter"
    assert char.race == "Human"
    # hp = max d8 + Con(14 -> +1) = 9
    assert char.hp_max == 9
    # ac = 11 + dex_mod(12 -> 0)
    assert char.ac == 11
    assert char.saves["death_ray"] == 12
    assert char.stats.attack_bonus == 1
    assert char.starting_gold % 10 == 0


def test_generate_character_unknown_class(classes_yaml: Path, races_yaml: Path) -> None:
    classes.load_classes(classes_yaml)
    races.load_races(races_yaml)
    abilities = AbilityScores(
        str_score=10,
        int_score=10,
        wis_score=10,
        dex_score=10,
        con_score=10,
        cha_score=10,
    )
    rng = random.Random(0)
    with pytest.raises(KeyError):
        generate_character(
            name="x",
            race_name="Human",
            class_name="Bard",
            alignment="lawful",
            rng=rng,
            abilities=abilities,
        )


def test_generate_character_class_disallowed_by_race(classes_yaml: Path, races_yaml: Path) -> None:
    """Human/Dwarf in fixture allow only Fighter/Thief; Cleric exists but is disallowed."""

    classes.load_classes(classes_yaml)
    races.load_races(races_yaml)
    abilities = AbilityScores(
        str_score=10,
        int_score=10,
        wis_score=10,
        dex_score=10,
        con_score=10,
        cha_score=10,
    )
    rng = random.Random(0)
    with pytest.raises(ValueError, match="cannot take class"):
        generate_character(
            name="x",
            race_name="Human",
            class_name="Cleric",
            alignment="lawful",
            rng=rng,
            abilities=abilities,
        )


def test_generate_character_dwarf_alignment_check(classes_yaml: Path, races_yaml: Path) -> None:
    classes.load_classes(classes_yaml)
    races.load_races(races_yaml)
    abilities = AbilityScores(
        str_score=12,
        int_score=10,
        wis_score=10,
        dex_score=10,
        con_score=12,
        cha_score=10,
    )
    rng = random.Random(0)
    # Dwarf disallows chaotic alignment in the fixture
    with pytest.raises(ValueError, match="alignment"):
        generate_character(
            name="x",
            race_name="Dwarf",
            class_name="Fighter",
            alignment="chaotic",
            rng=rng,
            abilities=abilities,
        )


def test_generate_character_dwarf_max_requirement(classes_yaml: Path, races_yaml: Path) -> None:
    """Dwarf has cha_max=17 in the fixture; cha=18 should fail."""

    classes.load_classes(classes_yaml)
    races.load_races(races_yaml)
    abilities = AbilityScores(
        str_score=12,
        int_score=10,
        wis_score=10,
        dex_score=10,
        con_score=12,
        cha_score=18,  # exceeds dwarf cha max 17
    )
    rng = random.Random(0)
    with pytest.raises(ValueError, match="max"):
        generate_character(
            name="x",
            race_name="Dwarf",
            class_name="Fighter",
            alignment="lawful",
            rng=rng,
            abilities=abilities,
        )


def test_generate_character_dwarf_con_requirement(classes_yaml: Path, races_yaml: Path) -> None:
    classes.load_classes(classes_yaml)
    races.load_races(races_yaml)
    abilities = AbilityScores(
        str_score=12,
        int_score=10,
        wis_score=10,
        dex_score=10,
        con_score=8,  # below dwarf's con min of 9
        cha_score=10,
    )
    rng = random.Random(0)
    with pytest.raises(ValueError, match="requirements"):
        generate_character(
            name="x",
            race_name="Dwarf",
            class_name="Fighter",
            alignment="lawful",
            rng=rng,
            abilities=abilities,
        )


def test_generate_character_inherits_save_modifiers(classes_yaml: Path, races_yaml: Path) -> None:
    classes.load_classes(classes_yaml)
    races.load_races(races_yaml)
    abilities = AbilityScores(
        str_score=12,
        int_score=10,
        wis_score=10,
        dex_score=10,
        con_score=14,
        cha_score=10,
    )
    rng = random.Random(0)
    char = generate_character(
        name="Thord",
        race_name="Dwarf",
        class_name="Fighter",
        alignment="lawful",
        rng=rng,
        abilities=abilities,
    )
    assert char.save_modifiers["magic_wand"] == 4
    assert char.save_modifiers["paralysis"] == 4
    # Stats inherits the modifiers too
    assert char.stats.save_modifiers["magic_wand"] == 4


def test_generate_character_rolls_abilities_when_omitted(
    classes_yaml: Path, races_yaml: Path
) -> None:
    classes.load_classes(classes_yaml)
    races.load_races(races_yaml)
    rng = random.Random(0)
    char = generate_character(
        name="x",
        race_name="Human",
        class_name="Fighter",
        alignment="lawful",
        rng=rng,
        method="heroic",
    )
    for ability in ("str", "int", "wis", "dex", "con", "cha"):
        assert 3 <= char.abilities.get(ability) <= 18


def test_ability_scores_get_helper() -> None:
    s = AbilityScores(
        str_score=11,
        int_score=12,
        wis_score=13,
        dex_score=14,
        con_score=15,
        cha_score=16,
    )
    assert s.get("str") == 11
    assert s.get("cha") == 16
