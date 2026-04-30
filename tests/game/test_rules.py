"""Tests for ``app.game.rules`` — the BFRPG resolver surface."""

from __future__ import annotations

import dataclasses
import random

import pytest

from app.game.rules import (
    CharacterStats,
    Participant,
    WeaponSpec,
    ability_check,
    ability_modifier,
    apply_damage,
    attack_roll,
    encounter_xp,
    heal,
    hp_at_level_up,
    roll_initiative,
    saving_throw,
    xp_for_treasure,
)

# --- Ability modifier curve ---------------------------------------------------


@pytest.mark.parametrize(
    "score,expected",
    [
        (3, -3),
        (4, -2),
        (5, -2),
        (6, -1),
        (8, -1),
        (9, 0),
        (12, 0),
        (13, 1),
        (15, 1),
        (16, 2),
        (17, 2),
        (18, 3),
    ],
)
def test_bfrpg_modifier_curve(score: int, expected: int) -> None:
    assert ability_modifier(score) == expected


def test_ability_modifier_extrapolates_below_three() -> None:
    # Curve continues: scores 1-2 -> -4, scores -1..0 -> -5, etc.
    assert ability_modifier(2) == -4
    assert ability_modifier(1) == -4
    assert ability_modifier(0) == -5
    assert ability_modifier(-1) == -5


def test_ability_modifier_extrapolates_above_eighteen() -> None:
    assert ability_modifier(19) == 4
    assert ability_modifier(20) == 4
    assert ability_modifier(21) == 5
    assert ability_modifier(22) == 5


# --- Helpers -------------------------------------------------------------------


def _make_fighter(
    *, str_score: int = 13, dex_score: int = 10, con_score: int = 10
) -> CharacterStats:
    return CharacterStats(
        name="Hrok",
        class_name="Fighter",
        level=1,
        hp_current=8,
        hp_max=8,
        ac=12,
        str_score=str_score,
        int_score=10,
        wis_score=10,
        dex_score=dex_score,
        con_score=con_score,
        cha_score=10,
        attack_bonus=1,
        saves={
            "death_ray": 12,
            "magic_wand": 13,
            "paralysis": 14,
            "dragon_breath": 15,
            "spells": 17,
        },
        save_modifiers={},
        hit_die="d8",
    )


# --- Attack roll ---------------------------------------------------------------


def test_natural_20_hits_regardless_of_ac() -> None:
    rng = random.Random(5)  # first d20 = 20
    attacker = _make_fighter()
    weapon = WeaponSpec(name="longsword", damage="1d8")
    result = attack_roll(attacker, target_ac=99, weapon=weapon, rng=rng)
    assert result.hit is True
    assert result.natural_twenty is True
    assert result.attack_roll.individual == [20]
    assert result.damage > 0
    assert result.damage_roll is not None


def test_natural_1_misses_regardless_of_bonus() -> None:
    rng = random.Random(31)  # first d20 = 1
    attacker = _make_fighter(str_score=18)
    weapon = WeaponSpec(name="longsword", damage="1d8")
    result = attack_roll(attacker, target_ac=2, weapon=weapon, rng=rng)
    assert result.hit is False
    assert result.natural_one is True
    assert result.damage == 0
    assert result.damage_roll is None


def test_attack_modifier_includes_str_for_melee() -> None:
    rng = random.Random(42)  # d20 = 4, then dice for damage
    attacker = _make_fighter(str_score=18)  # +3
    weapon = WeaponSpec(name="longsword", damage="1d8")
    # Total = 4 + 1(class) + 3(str) = 8, vs AC 10 -> miss
    result = attack_roll(attacker, target_ac=10, weapon=weapon, rng=rng)
    assert result.hit is False
    # Now vs AC 8 -> hit, damage rolled with +3 str
    rng = random.Random(42)
    result = attack_roll(attacker, target_ac=8, weapon=weapon, rng=rng)
    assert result.hit is True
    # damage_roll.total is 1d8+3
    assert result.damage_roll is not None
    assert result.damage_roll.total >= 4


def test_attack_uses_dex_modifier_for_ranged() -> None:
    rng = random.Random(42)  # d20 = 4
    attacker = _make_fighter(str_score=8, dex_score=18)  # str -1, dex +3
    weapon = WeaponSpec(name="shortbow", damage="1d6", is_ranged=True)
    # 4 + 1 + 3 (dex) = 8 vs AC 8 -> hit; damage gets no bonus on ranged
    result = attack_roll(attacker, target_ac=8, weapon=weapon, rng=rng)
    assert result.hit is True
    # damage roll is plain 1d6 (no bonus)
    assert result.damage_roll is not None
    assert result.damage_roll.expression == "1d6"


def test_attack_damage_floor_of_one() -> None:
    rng = random.Random(42)  # d20 = 4 -> hit at low AC
    attacker = _make_fighter(str_score=3)  # -3 str
    # 1d4 - 3, minimum 1: even on lowest die (1) damage is 1, not -2.
    weapon = WeaponSpec(name="dagger", damage="1d4")
    result = attack_roll(attacker, target_ac=2, weapon=weapon, rng=rng)
    assert result.hit is True
    assert result.damage >= 1


# --- Saving throws ------------------------------------------------------------


def test_save_succeeds_when_roll_meets_target() -> None:
    rng = random.Random(1)  # first d20 = 5
    char = dataclasses.replace(_make_fighter(), save_modifiers={"paralysis": 10})
    # 5 + 10 = 15, target 14 -> success
    result = saving_throw(char, "paralysis", rng=rng)
    assert result.success is True
    assert result.modifier == 10
    assert result.target == 14


def test_save_fails_below_target() -> None:
    rng = random.Random(42)  # first d20 = 4
    char = _make_fighter()
    result = saving_throw(char, "paralysis", rng=rng)
    # 4 + 0 = 4, target 14 -> fail
    assert result.success is False


def test_save_explicit_target_overrides_table() -> None:
    rng = random.Random(1)  # d20 = 5
    char = _make_fighter()
    result = saving_throw(char, "paralysis", target=4, rng=rng)
    assert result.success is True
    assert result.target == 4


def test_save_missing_table_raises_without_explicit_target() -> None:
    rng = random.Random(0)
    char = CharacterStats(
        name="Bare",
        class_name="None",
        level=1,
        hp_current=1,
        hp_max=1,
        ac=10,
        str_score=10,
        int_score=10,
        wis_score=10,
        dex_score=10,
        con_score=10,
        cha_score=10,
        saves={},
    )
    with pytest.raises(ValueError, match="no save target"):
        saving_throw(char, "paralysis", rng=rng)


def test_save_boundary_equals_target_succeeds() -> None:
    # d20 of exactly 14 with no mod: success against target 14
    rng = random.Random(0)
    # Find a seed with first d20 == 14
    for s in range(2000):
        r = random.Random(s)
        if r.randint(1, 20) == 14:
            rng = random.Random(s)
            break
    char = _make_fighter()
    result = saving_throw(char, "paralysis", rng=rng)
    assert result.roll.total == 14
    assert result.target == 14
    assert result.success is True


# --- Ability check ------------------------------------------------------------


def test_ability_check_roll_under_succeeds_at_score() -> None:
    # rolling under a high score should be easy
    rng = random.Random(42)  # d20 = 4
    char = _make_fighter(str_score=15)
    result = ability_check(char, "str", rng=rng)
    assert result.mode == "roll_under"
    assert result.success is True  # 4 <= 15


def test_ability_check_roll_under_fails_when_above_score() -> None:
    rng = random.Random(1)  # second d20=19, first=5; here first roll=5
    char = _make_fighter(str_score=4)
    # We need a roll > 4: seed=1 first d20=5, score=4, 5 > 4 -> fail
    result = ability_check(char, "str", rng=rng)
    assert result.success is False


def test_ability_check_roll_under_natural_1_always_succeeds() -> None:
    rng = random.Random(31)  # first d20=1
    char = _make_fighter(str_score=3)
    # Even with score 3, natural 1 is success in roll-under mode
    result = ability_check(char, "str", rng=rng)
    assert result.success is True
    assert result.roll.natural_one is True


def test_ability_check_roll_under_natural_20_always_fails() -> None:
    rng = random.Random(5)  # first d20=20
    char = _make_fighter(str_score=18)
    result = ability_check(char, "str", rng=rng)
    assert result.success is False
    assert result.roll.natural_twenty is True


def test_ability_check_dc_mode_with_modifier() -> None:
    rng = random.Random(1)  # d20 = 5
    char = _make_fighter(str_score=18)  # +3
    # 5 + 3 = 8, dc 8 -> success
    result = ability_check(char, "str", dc=8, rng=rng)
    assert result.mode == "dc"
    assert result.success is True
    assert result.dc == 8

    # dc 9 -> fail
    rng = random.Random(1)
    result = ability_check(char, "str", dc=9, rng=rng)
    assert result.success is False


def test_ability_check_each_ability_pulled_from_correct_field() -> None:
    rng = random.Random(0)
    char = CharacterStats(
        name="t",
        class_name="t",
        level=1,
        hp_current=1,
        hp_max=1,
        ac=10,
        str_score=11,
        int_score=12,
        wis_score=13,
        dex_score=14,
        con_score=15,
        cha_score=16,
    )
    for ability, score in [
        ("str", 11),
        ("int", 12),
        ("wis", 13),
        ("dex", 14),
        ("con", 15),
        ("cha", 16),
    ]:
        rng = random.Random(0)
        result = ability_check(char, ability, rng=rng)  # type: ignore[arg-type]
        assert result.score == score
        assert result.ability == ability


# --- Damage / heal ------------------------------------------------------------


def test_apply_damage_reduces_hp() -> None:
    char = _make_fighter()
    # hp_current=8 by default
    result = apply_damage(char, 3, source="goblin scimitar")
    assert result.previous_hp == 8
    assert result.new_hp == 5
    assert result.dropped_to_zero is False
    assert result.below_zero_by == 0


def test_apply_damage_zero_amount() -> None:
    char = _make_fighter()
    result = apply_damage(char, 0)
    assert result.new_hp == char.hp_current


def test_apply_damage_to_zero() -> None:
    char = _make_fighter()  # hp 8
    result = apply_damage(char, 8)
    assert result.new_hp == 0
    assert result.dropped_to_zero is True
    assert result.below_zero_by == 0


def test_apply_damage_below_zero() -> None:
    char = _make_fighter()  # hp 8
    result = apply_damage(char, 12)
    assert result.new_hp == -4
    assert result.dropped_to_zero is True
    assert result.below_zero_by == 4


def test_apply_damage_negative_amount_rejected() -> None:
    char = _make_fighter()
    with pytest.raises(ValueError, match="non-negative"):
        apply_damage(char, -5)


def test_heal_caps_at_hp_max() -> None:
    char = CharacterStats(
        name="t",
        class_name="t",
        level=1,
        hp_current=5,
        hp_max=8,
        ac=10,
        str_score=10,
        int_score=10,
        wis_score=10,
        dex_score=10,
        con_score=10,
        cha_score=10,
    )
    result = heal(char, 99)
    assert result.new_hp == 8
    assert result.previous_hp == 5


def test_heal_zero_or_partial() -> None:
    char = CharacterStats(
        name="t",
        class_name="t",
        level=1,
        hp_current=3,
        hp_max=8,
        ac=10,
        str_score=10,
        int_score=10,
        wis_score=10,
        dex_score=10,
        con_score=10,
        cha_score=10,
    )
    assert heal(char, 0).new_hp == 3
    assert heal(char, 4).new_hp == 7


def test_heal_negative_rejected() -> None:
    char = _make_fighter()
    with pytest.raises(ValueError, match="non-negative"):
        heal(char, -3)


def test_heal_zero_hp_rejected() -> None:
    char = CharacterStats(
        name="t",
        class_name="t",
        level=1,
        hp_current=0,
        hp_max=8,
        ac=10,
        str_score=10,
        int_score=10,
        wis_score=10,
        dex_score=10,
        con_score=10,
        cha_score=10,
    )
    with pytest.raises(ValueError, match="revival"):
        heal(char, 1)


# --- Level-up HP --------------------------------------------------------------


def test_hp_at_level_up_minimum_one_with_negative_con() -> None:
    char = CharacterStats(
        name="t",
        class_name="t",
        level=1,
        hp_current=1,
        hp_max=1,
        ac=10,
        str_score=10,
        int_score=10,
        wis_score=10,
        dex_score=10,
        con_score=3,  # -3 con mod
        cha_score=10,
        hit_die="d4",
    )
    # Even worst d4 roll (1) + (-3) = -2, but minimum is 1
    rng = random.Random(0)
    for _ in range(50):
        gained = hp_at_level_up(char, rng=rng)
        assert gained >= 1


def test_hp_at_level_up_uses_class_hit_die() -> None:
    char = _make_fighter(con_score=10)
    # d8: range 1..8 with +0 mod
    rng = random.Random(0)
    gained = hp_at_level_up(char, rng=rng)
    assert 1 <= gained <= 8


def test_hp_at_level_up_adds_con() -> None:
    char = _make_fighter(con_score=18)  # +3
    # min die 1 -> 1 + 3 = 4; max 8 -> 11
    rng = random.Random(0)
    for _ in range(50):
        rng2 = random.Random(rng.randrange(1 << 30))
        gained = hp_at_level_up(char, rng=rng2)
        assert 4 <= gained <= 11


# --- XP -----------------------------------------------------------------------


def test_xp_for_treasure_one_to_one() -> None:
    assert xp_for_treasure(0) == 0
    assert xp_for_treasure(50) == 50
    assert xp_for_treasure(1234) == 1234


def test_xp_for_treasure_negative_rejected() -> None:
    with pytest.raises(ValueError):
        xp_for_treasure(-1)


def test_encounter_xp_sums(monsters_yaml) -> None:  # type: ignore[no-untyped-def]
    # Use the fixture file
    from app.game import monsters as mons

    table = mons.load_monsters(monsters_yaml)
    goblin = table["Goblin"]
    skeleton = table["Skeleton"]
    assert encounter_xp([goblin, skeleton]) == 35
    assert encounter_xp([]) == 0
    assert encounter_xp([goblin, goblin, goblin]) == 30


# --- Initiative ---------------------------------------------------------------


def test_roll_initiative_orders_descending() -> None:
    rng = random.Random(0)
    participants = [
        Participant(participant_id="b", name="Bart", dex_modifier=2, is_player=True),
        Participant(participant_id="a", name="Alice", dex_modifier=0, is_player=True),
        Participant(participant_id="c", name="Goblin", dex_modifier=-1, is_player=False),
    ]
    order = roll_initiative(participants, rng=rng)
    assert len(order.entries) == 3
    # Ensure descending by initiative
    inits = [e.initiative for e in order.entries]
    assert inits == sorted(inits, reverse=True)
    assert order.round_number == 1
    assert order.index == 0


def test_initiative_ties_broken_deterministically_by_id() -> None:
    # Force a tie by using d1: every roll is 1.
    # Inject by mocking the dice expression — better, find a seed that
    # produces a tie naturally with two participants having the same dex
    # mod. Easier: use participants with dex_mod=0, find a seed that
    # produces equal d6 values for the first two participants.
    # We can also test by using a custom rng that always returns 1.
    class FixedRNG(random.Random):
        def randint(self, a: int, b: int) -> int:
            return a  # always lowest

    rng = FixedRNG()
    participants = [
        Participant(participant_id="zzz", name="Z", dex_modifier=0),
        Participant(participant_id="aaa", name="A", dex_modifier=0),
        Participant(participant_id="mmm", name="M", dex_modifier=0),
    ]
    order = roll_initiative(participants, rng=rng)
    # All tied at initiative 1, sorted by id ascending
    ids = [e.participant_id for e in order.entries]
    assert ids == ["aaa", "mmm", "zzz"]
