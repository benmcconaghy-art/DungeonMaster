"""Tests for ``app.game.races`` (loader + helpers)."""

from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import pytest
from pydantic import ValidationError

from app.game import races


def test_loader_indexes_by_name(races_yaml: Path) -> None:
    table = races.load_races(races_yaml)
    assert set(table) == {"Human", "Dwarf"}
    dwarf = table["Dwarf"]
    assert dwarf.movement == 30
    assert dwarf.infravision == 60
    assert dwarf.save_modifiers["magic_wand"] == 4


def test_get_race(races_yaml: Path) -> None:
    races.load_races(races_yaml)
    dwarf = races.get_race("Dwarf")
    assert dwarf.name == "Dwarf"


def test_unknown_race_raises(races_yaml: Path) -> None:
    races.load_races(races_yaml)
    with pytest.raises(KeyError):
        races.get_race("Drow")


def test_race_allows_class(races_yaml: Path) -> None:
    races.load_races(races_yaml)
    dwarf = races.get_race("Dwarf")
    assert races.race_allows_class(dwarf, "Fighter") is True
    assert races.race_allows_class(dwarf, "Magic-User") is False


def test_level_cap_uncapped_when_missing(races_yaml: Path) -> None:
    races.load_races(races_yaml)
    human = races.get_race("Human")
    # Human has empty level_caps but allows Fighter -> uncapped
    assert races.level_cap_for(human, "Fighter") is None


def test_level_cap_capped_for_dwarf(races_yaml: Path) -> None:
    races.load_races(races_yaml)
    dwarf = races.get_race("Dwarf")
    assert races.level_cap_for(dwarf, "Fighter") == 8
    assert races.level_cap_for(dwarf, "Thief") == 10


def test_level_cap_for_disallowed_class_raises(races_yaml: Path) -> None:
    races.load_races(races_yaml)
    dwarf = races.get_race("Dwarf")
    with pytest.raises(ValueError):
        races.level_cap_for(dwarf, "Magic-User")


def test_top_level_must_be_list(tmp_path: Path) -> None:
    bad = tmp_path / "races.yaml"
    bad.write_text("name: not-a-list", encoding="utf-8")
    with pytest.raises(ValueError):
        races.load_races(bad)


def test_extra_field_rejected(tmp_path: Path) -> None:
    bad = tmp_path / "races.yaml"
    bad.write_text(
        dedent(
            """\
            - name: Goblinoid
              allowed_classes: [Fighter]
              allowed_alignments: [chaotic]
              extra_field: 1
            """
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValidationError):
        races.load_races(bad)


def test_reset_cache(races_yaml: Path) -> None:
    races.load_races(races_yaml)
    races.reset_cache()
    # Without a path the loader would look for the default path; pass our fixture.
    table = races.load_races(races_yaml)
    assert "Dwarf" in table


def test_explicit_null_level_cap(tmp_path: Path) -> None:
    """``level_caps: {ClassName: null}`` means uncapped."""

    payload = dedent(
        """\
        - name: Elf
          allowed_classes: [Fighter, Magic-User]
          level_caps: {Fighter: 4, Magic-User: null}
          allowed_alignments: [neutral]
        """
    )
    p = tmp_path / "races.yaml"
    p.write_text(payload, encoding="utf-8")
    table = races.load_races(p)
    elf = table["Elf"]
    assert races.level_cap_for(elf, "Fighter") == 4
    assert races.level_cap_for(elf, "Magic-User") is None
