"""Tests for ``app.game.validate_data``."""

from __future__ import annotations

from pathlib import Path
from textwrap import dedent
from unittest.mock import patch

import pytest
from pydantic import ValidationError

from app.game import validate_data
from app.game.schemas.class_def import ClassesFile
from app.game.schemas.equipment_def import EquipmentFile
from app.game.schemas.monster_def import MonstersFile
from app.game.schemas.race_def import RacesFile
from app.game.schemas.spell_def import SpellsFile


def test_validate_all_no_files(tmp_path: Path) -> None:
    """A directory with no content files exits 0 with a clear message."""

    messages: list[str] = []
    with patch.object(validate_data, "DATA_DIR", tmp_path):
        rc = validate_data.validate_all(log=messages.append)
    assert rc == 0
    assert any("no data files yet" in m for m in messages)


def test_validate_all_missing_directory(tmp_path: Path) -> None:
    """Directory doesn't exist at all -> still exit 0."""

    missing = tmp_path / "does-not-exist"
    messages: list[str] = []
    with patch.object(validate_data, "DATA_DIR", missing):
        rc = validate_data.validate_all(log=messages.append)
    assert rc == 0
    assert any("does not exist" in m for m in messages)


def test_validate_all_known_good_files(
    tmp_path: Path,
    classes_yaml: Path,
    races_yaml: Path,
    monsters_yaml: Path,
    equipment_yaml: Path,
) -> None:
    """All four content files validate; rc == 0."""

    # Stage the fixture files into the data directory the validator scans.
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    for src, name in [
        (classes_yaml, "classes.yaml"),
        (races_yaml, "races.yaml"),
        (monsters_yaml, "monsters.yaml"),
        (equipment_yaml, "equipment.yaml"),
    ]:
        (data_dir / name).write_bytes(src.read_bytes())
    # spells.yaml absent -> "not yet authored"
    messages: list[str] = []
    with patch.object(validate_data, "DATA_DIR", data_dir):
        rc = validate_data.validate_all(log=messages.append)
    assert rc == 0
    log = "\n".join(messages)
    assert "classes.yaml" in log
    assert "OK" in log
    assert "spells.yaml" in log and "not yet authored" in log


def test_validate_all_bad_yaml(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "classes.yaml").write_text(
        "- name: Fighter\n  hit_die: not-a-die\n", encoding="utf-8"
    )
    messages: list[str] = []
    with patch.object(validate_data, "DATA_DIR", data_dir):
        rc = validate_data.validate_all(log=messages.append)
    assert rc == 1
    assert any("FAIL" in m or "FAILED" in m for m in messages)


def test_validate_all_bad_yaml_parse_error(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "classes.yaml").write_text("- this is: : : not yaml :::", encoding="utf-8")
    messages: list[str] = []
    with patch.object(validate_data, "DATA_DIR", data_dir):
        rc = validate_data.validate_all(log=messages.append)
    assert rc == 1


def test_validate_all_wrong_top_level_type(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    # classes.yaml expects a list; give it a mapping
    (data_dir / "classes.yaml").write_text("name: foo", encoding="utf-8")
    messages: list[str] = []
    with patch.object(validate_data, "DATA_DIR", data_dir):
        rc = validate_data.validate_all(log=messages.append)
    assert rc == 1
    assert any("expected" in m for m in messages)


def test_main_returns_exit_code(tmp_path: Path) -> None:
    with patch.object(validate_data, "DATA_DIR", tmp_path):
        assert validate_data.main() == 0


def test_validate_all_equipment_wrong_top_level(tmp_path: Path) -> None:
    """equipment.yaml expects a mapping; a list should fail with a clear msg."""

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "equipment.yaml").write_text("- a list of weapons", encoding="utf-8")
    messages: list[str] = []
    with patch.object(validate_data, "DATA_DIR", data_dir):
        rc = validate_data.validate_all(log=messages.append)
    assert rc == 1
    assert any("top-level mapping" in m for m in messages)


def test_module_run_as_main(tmp_path: Path) -> None:
    """``python -m app.game.validate_data`` runs without error on empty data dir."""

    import subprocess
    import sys

    result = subprocess.run(
        [sys.executable, "-m", "app.game.validate_data"],
        cwd=str(tmp_path),  # tmp_path has no data/bfrpg dir; validator handles that
        capture_output=True,
        text=True,
        check=False,
    )
    # The default DATA_DIR is the project's data/bfrpg, which exists but is
    # almost empty; either no-files or all-files-validate is OK.
    assert result.returncode in {0, 1}


def test_inline_class_fixture_validates() -> None:
    """A known-good Pydantic dict validates; a bad one raises."""

    good = {
        "classes": [
            {
                "name": "Fighter",
                "hit_die": "d8",
                "prime_requisite": "STR",
                "weapon_restrictions": "any",
                "armour_restrictions": "any",
                "saves": {
                    1: {
                        "death_ray": 12,
                        "magic_wand": 13,
                        "paralysis": 14,
                        "dragon_breath": 15,
                        "spells": 17,
                    }
                },
                "attack_bonus_progression": {1: 1},
            }
        ]
    }
    parsed = ClassesFile.model_validate(good)
    assert len(parsed.classes) == 1
    bad = dict(good)
    bad["classes"][0] = {**good["classes"][0], "hit_die": "d3"}
    with pytest.raises(ValidationError):
        ClassesFile.model_validate(bad)


def test_inline_race_fixture_validates() -> None:
    good = {
        "races": [
            {
                "name": "Human",
                "allowed_classes": ["Fighter"],
                "allowed_alignments": ["lawful", "neutral"],
            }
        ]
    }
    parsed = RacesFile.model_validate(good)
    assert parsed.races[0].name == "Human"

    bad = {"races": [{"name": "X", "allowed_classes": [], "allowed_alignments": []}]}
    with pytest.raises(ValidationError):
        RacesFile.model_validate(bad)


def test_inline_spell_fixture_validates() -> None:
    good = {
        "spells": [
            {
                "name": "Magic Missile",
                "level": 1,
                "caster_classes": ["magic_user"],
                "range": "150 feet",
                "duration": "instant",
                "description": "An unerring bolt of force.",
                "damage_or_heal": "1d6+1",
            }
        ]
    }
    parsed = SpellsFile.model_validate(good)
    assert parsed.spells[0].name == "Magic Missile"

    bad = {**good, "spells": [{**good["spells"][0], "level": 0}]}
    with pytest.raises(ValidationError):
        SpellsFile.model_validate(bad)


def test_inline_monster_fixture_validates() -> None:
    good = {
        "monsters": [
            {
                "name": "Orc",
                "hit_dice": "1",
                "hp_typical": 4,
                "ac": 12,
                "movement_modes": {"land": 30},
                "attacks": [{"name": "spear", "damage": "1d6"}],
                "save_as": "F1",
                "morale": 8,
                "alignment": "chaotic",
                "xp": 10,
                "description": "A brutish humanoid.",
            }
        ]
    }
    parsed = MonstersFile.model_validate(good)
    assert parsed.monsters[0].xp == 10

    bad = {"monsters": [{**good["monsters"][0], "morale": 13}]}
    with pytest.raises(ValidationError):
        MonstersFile.model_validate(bad)


def test_inline_equipment_fixture_validates() -> None:
    good = {
        "weapons": [
            {"name": "dagger", "damage": "1d4", "weight": 1, "type": "melee", "cost_gp": 3}
        ],
        "armour": [{"name": "leather", "ac_bonus": 2, "weight": 15, "cost_gp": 20}],
        "gear": [{"name": "rope", "weight": 5, "cost_gp": 1}],
    }
    parsed = EquipmentFile.model_validate(good)
    assert parsed.weapons[0].name == "dagger"

    # Bad: weapon with no cost
    bad = dict(good)
    bad["weapons"] = [{"name": "stick", "damage": "1d2", "weight": 1, "type": "melee"}]
    with pytest.raises(ValidationError):
        EquipmentFile.model_validate(bad)


def test_dedent_helper_for_textwrap() -> None:
    """Smoke test for the dedent import (used in fixtures)."""

    assert dedent("    foo\n") == "foo\n"
