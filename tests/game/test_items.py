"""Tests for ``app.game.items`` (equipment loader)."""

from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import pytest
from pydantic import ValidationError

from app.game import items


def test_loader_populates_three_tables(equipment_yaml: Path) -> None:
    weapons = items.weapons(equipment_yaml)
    armour = items.armour(equipment_yaml)
    gear = items.gear(equipment_yaml)
    assert "longsword" in weapons
    assert "leather" in armour
    assert "torch" in gear


def test_get_weapon_armor_gear(equipment_yaml: Path) -> None:
    items.weapons(equipment_yaml)  # prime cache
    longsword = items.get_weapon("longsword")
    assert longsword.damage == "1d8"
    leather = items.get_armor("leather")
    assert leather.ac_bonus == 2
    torch = items.get_gear("torch")
    assert torch.cost_sp == 1


def test_unknown_item_raises(equipment_yaml: Path) -> None:
    items.weapons(equipment_yaml)
    with pytest.raises(KeyError):
        items.get_weapon("vorpal blade")
    with pytest.raises(KeyError):
        items.get_armor("mithril mail")
    with pytest.raises(KeyError):
        items.get_gear("zippo lighter")


def test_top_level_must_be_mapping(tmp_path: Path) -> None:
    bad = tmp_path / "equipment.yaml"
    bad.write_text("- a list", encoding="utf-8")
    with pytest.raises(ValueError, match="top-level mapping"):
        items.weapons(bad)


def test_extra_field_rejected(tmp_path: Path) -> None:
    bad = tmp_path / "equipment.yaml"
    bad.write_text(
        dedent(
            """\
            weapons:
              - {name: dagger, damage: "1d4", weight: 1, type: melee, cost_gp: 3, made_up: yes}
            armour: []
            """
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValidationError):
        items.weapons(bad)


def test_cost_required(tmp_path: Path) -> None:
    bad = tmp_path / "equipment.yaml"
    bad.write_text(
        dedent(
            """\
            weapons:
              - {name: stick, damage: "1d2", weight: 1, type: melee}
            armour:
              - {name: cloth, ac_bonus: 1, weight: 5, cost_gp: 1}
            """
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValidationError):
        items.weapons(bad)


def test_reset_cache(equipment_yaml: Path) -> None:
    items.weapons(equipment_yaml)
    assert items.weapons() == items._weapons_cache  # cache is shared
    items.reset_cache()
    assert items._weapons_cache == {}
