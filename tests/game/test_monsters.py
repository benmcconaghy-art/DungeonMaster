"""Tests for ``app.game.monsters``."""

from __future__ import annotations

from pathlib import Path

import pytest

from app.game import monsters


def test_loader_indexes_by_name(monsters_yaml: Path) -> None:
    table = monsters.load_monsters(monsters_yaml)
    assert "Goblin" in table
    goblin = table["Goblin"]
    assert goblin.ac == 14
    assert goblin.xp == 10
    assert goblin.attacks[0].damage == "1d6"


def test_get_monster(monsters_yaml: Path) -> None:
    monsters.load_monsters(monsters_yaml)
    skeleton = monsters.get_monster("Skeleton")
    assert skeleton.hit_dice == "1"


def test_unknown_monster_raises(monsters_yaml: Path) -> None:
    monsters.load_monsters(monsters_yaml)
    with pytest.raises(KeyError):
        monsters.get_monster("Tarrasque")


def test_top_level_must_be_list(tmp_path: Path) -> None:
    bad = tmp_path / "monsters.yaml"
    bad.write_text("name: not-a-list", encoding="utf-8")
    with pytest.raises(ValueError):
        monsters.load_monsters(bad)


@pytest.mark.parametrize(
    "notation,expected",
    [
        ("1", (1, 0)),
        ("2", (2, 0)),
        ("1+1", (1, 1)),
        ("1-1", (1, -1)),
        ("3+2", (3, 2)),
        ("½", (0, 0)),
        ("1/2", (0, 0)),
    ],
)
def test_parse_hit_dice(notation: str, expected: tuple[int, int]) -> None:
    assert monsters.parse_hit_dice(notation) == expected


def test_reset_cache(monsters_yaml: Path) -> None:
    monsters.load_monsters(monsters_yaml)
    monsters.reset_cache()
    assert monsters.load_monsters(monsters_yaml)["Goblin"]
