"""Loader for ``data/bfrpg/equipment.yaml``.

Exposes three indices — :func:`weapons`, :func:`armour`, :func:`gear` —
each keyed by item name. The Pydantic models from
:mod:`app.game.schemas.equipment_def` (``WeaponDefinition``,
``ArmorDefinition``, ``GearDefinition``) are re-exported as the
canonical engine types under the friendlier aliases ``Weapon``,
``Armor``, ``Gear`` so call sites read naturally::

    from app.game.items import Weapon, get_weapon
"""

from __future__ import annotations

from pathlib import Path

import yaml

from app.game.schemas.equipment_def import (
    ArmorDefinition,
    EquipmentFile,
    GearDefinition,
    WeaponDefinition,
)

# Re-exports — the loader's external vocabulary.
Weapon = WeaponDefinition
Armor = ArmorDefinition
Gear = GearDefinition

_DEFAULT_PATH = Path(__file__).resolve().parents[2] / "data" / "bfrpg" / "equipment.yaml"

_weapons_cache: dict[str, WeaponDefinition] = {}
_armour_cache: dict[str, ArmorDefinition] = {}
_gear_cache: dict[str, GearDefinition] = {}
_loaded_path: Path | None = None


def _load_from(path: Path) -> EquipmentFile:
    """Parse and validate ``path``; return the typed wrapper."""

    with path.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: expected a top-level mapping (weapons / armour / gear)")
    return EquipmentFile.model_validate(raw)


def _ensure_loaded(path: Path | None) -> None:
    """Populate the three caches if they're empty."""

    global _loaded_path
    if _weapons_cache or _armour_cache or _gear_cache:
        return
    target = path if path is not None else _DEFAULT_PATH
    parsed = _load_from(target)
    _weapons_cache.update({w.name: w for w in parsed.weapons})
    _armour_cache.update({a.name: a for a in parsed.armour})
    _gear_cache.update({g.name: g for g in parsed.gear})
    _loaded_path = target


def weapons(path: Path | None = None) -> dict[str, WeaponDefinition]:
    """Return the weapon table by name."""

    _ensure_loaded(path)
    return _weapons_cache


def armour(path: Path | None = None) -> dict[str, ArmorDefinition]:
    """Return the armour table by name."""

    _ensure_loaded(path)
    return _armour_cache


def gear(path: Path | None = None) -> dict[str, GearDefinition]:
    """Return the gear table by name."""

    _ensure_loaded(path)
    return _gear_cache


def get_weapon(name: str, *, path: Path | None = None) -> WeaponDefinition:
    """Lookup a weapon. Raises :class:`KeyError` if unknown."""

    table = weapons(path)
    try:
        return table[name]
    except KeyError as exc:
        raise KeyError(f"unknown weapon: {name!r}") from exc


def get_armor(name: str, *, path: Path | None = None) -> ArmorDefinition:
    """Lookup armour. Raises :class:`KeyError` if unknown."""

    table = armour(path)
    try:
        return table[name]
    except KeyError as exc:
        raise KeyError(f"unknown armour: {name!r}") from exc


def get_gear(name: str, *, path: Path | None = None) -> GearDefinition:
    """Lookup an ordinary gear item. Raises :class:`KeyError` if unknown."""

    table = gear(path)
    try:
        return table[name]
    except KeyError as exc:
        raise KeyError(f"unknown gear: {name!r}") from exc


def reset_cache() -> None:
    """Discard all three in-memory item tables."""

    global _loaded_path
    _weapons_cache.clear()
    _armour_cache.clear()
    _gear_cache.clear()
    _loaded_path = None
