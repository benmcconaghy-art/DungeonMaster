"""Loader for ``data/bfrpg/races.yaml``.

The schema is owned by this engine (``app.game.schemas.race_def``); see
that module's docstring for the layout rationale. Loader behaviour
mirrors :mod:`app.game.classes`: cache on first load, reset for tests
via :func:`reset_cache`.
"""

from __future__ import annotations

from pathlib import Path

import yaml

from app.game.schemas.race_def import RaceDefinition, RacesFile

_DEFAULT_PATH = Path(__file__).resolve().parents[2] / "data" / "bfrpg" / "races.yaml"

_races_cache: dict[str, RaceDefinition] = {}
_loaded_path: Path | None = None


def _load_from(path: Path) -> dict[str, RaceDefinition]:
    """Parse ``path`` (YAML) into the validated mapping by race name."""

    with path.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    if not isinstance(raw, list):
        raise ValueError(f"{path}: expected a top-level list of races")
    parsed = RacesFile.model_validate({"races": raw})
    return {definition.name: definition for definition in parsed.races}


def load_races(path: Path | None = None) -> dict[str, RaceDefinition]:
    """Return the cached race table, loading from ``path`` on first call."""

    global _loaded_path
    if not _races_cache:
        target = path if path is not None else _DEFAULT_PATH
        _races_cache.update(_load_from(target))
        _loaded_path = target
    return _races_cache


def get_race(name: str, *, path: Path | None = None) -> RaceDefinition:
    """Look up a race by name. Raises :class:`KeyError` if unknown."""

    table = load_races(path)
    try:
        return table[name]
    except KeyError as exc:
        raise KeyError(f"unknown race: {name!r} (known: {sorted(table)})") from exc


def race_allows_class(race: RaceDefinition, class_name: str) -> bool:
    """``True`` if ``class_name`` is in ``race.allowed_classes``."""

    return class_name in race.allowed_classes


def level_cap_for(race: RaceDefinition, class_name: str) -> int | None:
    """Return the per-class level cap, or ``None`` if uncapped.

    A class missing from ``level_caps`` but present in ``allowed_classes``
    is treated as uncapped — humans don't need an explicit table. A
    class missing from both raises :class:`ValueError`.
    """

    if class_name not in race.allowed_classes:
        raise ValueError(f"race {race.name!r} cannot take class {class_name!r}")
    if class_name not in race.level_caps:
        return None
    return race.level_caps[class_name]


def reset_cache() -> None:
    """Discard the in-memory race table. Tests call this between cases."""

    global _loaded_path
    _races_cache.clear()
    _loaded_path = None
