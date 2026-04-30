"""Shared fixtures for ``app.game.*`` tests.

The data-loader caches (``classes``, ``races``, ``items``, ``monsters``)
are module-level dicts, so a test that points one at a custom YAML
fixture would otherwise pollute every later test that uses the same
loader. The autouse ``_clear_engine_caches`` fixture wipes all four
between tests.

A pair of helper fixtures, ``classes_yaml`` and ``races_yaml``, write a
minimal-but-valid content file to a temp path and return it. Tests that
exercise the chargen path use those.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from textwrap import dedent

import pytest

from app.game import classes, items, monsters, races


@pytest.fixture(autouse=True)
def _clear_engine_caches() -> Iterator[None]:
    """Reset every loader cache before *and* after each test.

    Both directions matter — a previous test (or even module import
    side effects) might have populated the cache before this test
    starts; and this test might populate it for the next.
    """

    classes.reset_cache()
    races.reset_cache()
    items.reset_cache()
    monsters.reset_cache()
    yield
    classes.reset_cache()
    races.reset_cache()
    items.reset_cache()
    monsters.reset_cache()


_FIGHTER_AND_THIEF_YAML = dedent(
    """\
    - name: Fighter
      hit_die: d8
      prime_requisite: STR
      prime_req_bonus_threshold: 13
      weapon_restrictions: any
      armour_restrictions: any
      saves:
        1:  {death_ray: 12, magic_wand: 13, paralysis: 14, dragon_breath: 15, spells: 17}
        2:  {death_ray: 11, magic_wand: 12, paralysis: 13, dragon_breath: 14, spells: 16}
      attack_bonus_progression:
        1: 1
        2: 2
      special_abilities: []
    - name: Thief
      hit_die: d4
      prime_requisite: DEX
      prime_req_bonus_threshold: 13
      weapon_restrictions: limited
      armour_restrictions: leather only
      saves:
        1:  {death_ray: 13, magic_wand: 14, paralysis: 13, dragon_breath: 16, spells: 15}
      attack_bonus_progression:
        1: 1
      special_abilities: ["Backstab"]
      thief_skills:
        1:
          open_locks: 25
          remove_traps: 20
          pick_pockets: 30
          move_silently: 20
          climb_walls: 80
          hide: 10
          listen: 30
    - name: Cleric
      hit_die: d6
      prime_requisite: WIS
      weapon_restrictions: blunt only
      armour_restrictions: any
      saves:
        1:  {death_ray: 11, magic_wand: 12, paralysis: 14, dragon_breath: 16, spells: 15}
      attack_bonus_progression:
        1: 1
      special_abilities: ["Turn Undead"]
    """
)


_FOUR_RACES_YAML = dedent(
    """\
    - name: Human
      description: Adaptable, ambitious, and the most populous race.
      allowed_classes: [Fighter, Thief]
      level_caps: {}
      save_modifiers: {}
      special_abilities: []
      allowed_alignments: [lawful, neutral, chaotic]
      languages: [common]
      movement: 40
      infravision: 0
      source: core
    - name: Dwarf
      description: Stout, gruff, magic-resistant mountainfolk.
      ability_requirements:
        con: {min: 9}
        cha: {max: 17}
      allowed_classes: [Fighter, Thief]
      level_caps: {Fighter: 8, Thief: 10}
      save_modifiers:
        magic_wand: 4
        paralysis: 4
        dragon_breath: 4
        spells: 4
      special_abilities: ["+1 to hit with axes"]
      allowed_alignments: [lawful, neutral]
      languages: [common, dwarvish]
      movement: 30
      infravision: 60
      source: core
    """
)


@pytest.fixture
def classes_yaml(tmp_path: Path) -> Path:
    """Write a fixture ``classes.yaml`` to ``tmp_path`` and return it."""

    path = tmp_path / "classes.yaml"
    path.write_text(_FIGHTER_AND_THIEF_YAML, encoding="utf-8")
    return path


@pytest.fixture
def races_yaml(tmp_path: Path) -> Path:
    """Write a fixture ``races.yaml`` to ``tmp_path`` and return it."""

    path = tmp_path / "races.yaml"
    path.write_text(_FOUR_RACES_YAML, encoding="utf-8")
    return path


@pytest.fixture
def equipment_yaml(tmp_path: Path) -> Path:
    """Write a minimal valid ``equipment.yaml`` to ``tmp_path``."""

    payload = dedent(
        """\
        weapons:
          - name: dagger
            damage: "1d4"
            weight: 1
            cost_gp: 3
            type: melee
          - name: longsword
            damage: "1d8"
            weight: 4
            cost_gp: 30
            type: melee
          - name: shortbow
            damage: "1d6"
            weight: 2
            cost_gp: 25
            type: ranged
            range: [50, 100, 150]
        armour:
          - name: leather
            ac_bonus: 2
            weight: 15
            cost_gp: 20
        gear:
          - name: torch
            weight: 1
            cost_sp: 1
        """
    )
    path = tmp_path / "equipment.yaml"
    path.write_text(payload, encoding="utf-8")
    return path


@pytest.fixture
def monsters_yaml(tmp_path: Path) -> Path:
    """Write a minimal valid ``monsters.yaml`` to ``tmp_path``."""

    payload = dedent(
        """\
        - name: Goblin
          hit_dice: "1-1"
          hp_typical: 3
          ac: 14
          movement_modes: {land: 20}
          attacks:
            - {name: scimitar, damage: "1d6", to_hit_bonus: 0}
          no_appearing: "2d4 (6d10)"
          save_as: F1
          morale: 7
          alignment: chaotic
          treasure_type: "(P) each, R lair"
          xp: 10
          description: A wiry, malnourished humanoid four feet tall.
        - name: Skeleton
          hit_dice: "1"
          hp_typical: 4
          ac: 12
          movement_modes: {land: 20}
          attacks:
            - {name: claw, damage: "1d6"}
          save_as: F1
          morale: 12
          alignment: chaotic
          xp: 25
          description: An animated humanoid skeleton.
        """
    )
    path = tmp_path / "monsters.yaml"
    path.write_text(payload, encoding="utf-8")
    return path
