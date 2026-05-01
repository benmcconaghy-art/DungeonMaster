"""Tests for ``app.game.dice``."""

from __future__ import annotations

import random

import pytest

from app.game.dice import Roll, roll


def test_simple_d20_uses_injected_rng() -> None:
    rng = random.Random(42)
    # seed=42, first randint(1,20) = 4
    result = roll("1d20", rng=rng)
    assert result.total == 4
    assert result.individual == [4]
    assert result.expression == "1d20"
    assert not result.natural_one
    assert not result.natural_twenty


def test_d20_default_count_one() -> None:
    rng = random.Random(42)
    result = roll("d20", rng=rng)
    assert result.individual == [4]
    assert result.total == 4


def test_natural_twenty_flagged() -> None:
    rng = random.Random(5)  # seed=5 first randint(1,20)=20
    result = roll("1d20", rng=rng)
    assert result.individual == [20]
    assert result.natural_twenty is True
    assert result.natural_one is False


def test_natural_one_flagged() -> None:
    rng = random.Random(31)  # seed=31 first randint(1,20)=1
    result = roll("1d20", rng=rng)
    assert result.individual == [1]
    assert result.natural_one is True
    assert result.natural_twenty is False


def test_d20_with_positive_modifier() -> None:
    rng = random.Random(42)
    result = roll("1d20+5", rng=rng)
    assert result.individual == [4]
    assert result.total == 9


def test_d20_with_negative_modifier() -> None:
    rng = random.Random(42)
    result = roll("1d20-3", rng=rng)
    assert result.total == 1


def test_3d6_sums() -> None:
    rng = random.Random(11)
    result = roll("3d6", rng=rng)
    assert len(result.individual) == 3
    assert result.total == sum(result.individual)
    # natural-1/20 only applies to 1d20 expressions
    assert not result.natural_one
    assert not result.natural_twenty


def test_4d6kh3_drops_lowest() -> None:
    rng = random.Random(11)  # 4 rolls of d6: [4,5,4,4] -> kh3 = [5,4,4] = 13
    result = roll("4d6kh3", rng=rng)
    assert (
        sorted(result.individual, reverse=True) == result.individual or len(result.individual) == 3
    )
    assert len(result.individual) == 3
    assert result.total == 13


def test_4d6kl1_keeps_lowest() -> None:
    rng = random.Random(11)  # rolls [4,5,4,4] -> kl1 = [4]
    result = roll("4d6kl1", rng=rng)
    assert result.individual == [4]
    assert result.total == 4


def test_2d20kh1_advantage_emulation() -> None:
    # advantage as keep-highest pair
    rng = random.Random(42)  # first two d20 = 4, 1
    result = roll("2d20kh1", rng=rng)
    assert result.individual == [4]
    assert result.total == 4
    # Note: this is keep-highest of N dice, not the advantage flag — so
    # the natural-1/20 helper does not light up (count==2).
    assert result.natural_twenty is False


def test_advantage_flag_picks_higher_of_two_rolls() -> None:
    rng = random.Random(42)  # first two d20s = 4, 1 -> advantage takes 4
    result = roll("1d20", advantage=True, rng=rng)
    assert result.individual == [4]
    assert result.total == 4


def test_disadvantage_flag_picks_lower_of_two_rolls() -> None:
    rng = random.Random(42)  # first two d20s = 4, 1 -> disadvantage takes 1
    result = roll("1d20", disadvantage=True, rng=rng)
    assert result.individual == [1]
    assert result.total == 1
    assert result.natural_one is True


def test_advantage_uses_alt_when_higher() -> None:
    # Use a seed where the second roll is strictly higher.
    # seed=1 first two d20s: [5, 19]
    rng = random.Random(1)
    result = roll("1d20", advantage=True, rng=rng)
    assert result.individual == [19]
    assert result.total == 19


def test_advantage_and_disadvantage_mutually_exclusive() -> None:
    rng = random.Random(0)
    with pytest.raises(ValueError, match="mutually exclusive"):
        roll("1d20", advantage=True, disadvantage=True, rng=rng)


@pytest.mark.parametrize(
    "expression",
    [
        "",
        "20",
        "d",
        "1d",
        "0d6",
        "2d0",
        "abc",
        "1d6+",
        "4d6kh5",  # keep > count
        "4d6kh0",
        "1d20kx2",
    ],
)
def test_invalid_expressions_raise(expression: str) -> None:
    rng = random.Random(0)
    with pytest.raises(ValueError):
        roll(expression, rng=rng)


def test_roll_dataclass_is_frozen() -> None:
    import dataclasses

    r = Roll(expression="1d6", total=3, individual=[3])
    with pytest.raises(dataclasses.FrozenInstanceError):
        r.total = 4  # type: ignore[misc]


def test_whitespace_tolerated() -> None:
    rng = random.Random(42)
    result = roll(" 1d20 + 5 ", rng=rng)
    assert result.total == 9


def test_advantage_keeps_first_when_equal() -> None:
    # If both advantage rolls produce the same total, the first is kept.
    # We can craft this by using a 1-sided die, where every roll is the same.
    rng = random.Random(0)
    result = roll("1d1", advantage=True, rng=rng)
    assert result.total == 1
    assert result.individual == [1]


# ---------------------------------------------------------------------------
# Multiplier — BFRPG starting-gold idiom (3d6*10) and friends
# ---------------------------------------------------------------------------


def test_multiplier_bfrpg_starting_gold() -> None:
    rng = random.Random(11)  # 3d6 with seed 11 -> [4, 5, 4] = 13
    result = roll("3d6*10", rng=rng)
    assert result.individual == [4, 5, 4]
    assert result.total == 130


def test_multiplier_with_spaces_around_star() -> None:
    rng = random.Random(11)
    result = roll("3d6 * 10", rng=rng)
    assert result.total == 130


def test_multiplier_with_parens_around_dice() -> None:
    rng = random.Random(11)
    result = roll("(3d6)*10", rng=rng)
    assert result.total == 130


def test_multiplier_combined_with_modifier() -> None:
    rng = random.Random(11)  # 3d6 -> [4, 5, 4] = 13; *10 + 5 = 135
    result = roll("3d6*10+5", rng=rng)
    assert result.total == 135


def test_multiplier_combined_with_negative_modifier() -> None:
    rng = random.Random(11)  # 13 * 10 - 3 = 127
    result = roll("3d6*10-3", rng=rng)
    assert result.total == 127


def test_modifier_alone_still_works() -> None:
    """Adding the multiplier rule must not break the bare modifier path."""

    rng = random.Random(42)  # 1d20 -> [4]; +1 -> 5
    result = roll("2d4+1", rng=rng)
    assert sum(result.individual) + 1 == result.total


def test_multiplier_must_be_positive() -> None:
    rng = random.Random(0)
    with pytest.raises(ValueError):
        roll("3d6*0", rng=rng)


@pytest.mark.parametrize(
    "expression",
    [
        "(3d6",  # unbalanced
        "3d6)",
        "3d6*",  # missing multiplier value
        "3d6*+5",
        "3d6**10",
    ],
)
def test_invalid_multiplier_expressions_raise(expression: str) -> None:
    rng = random.Random(0)
    with pytest.raises(ValueError):
        roll(expression, rng=rng)
