"""Loader for ``data/bfrpg/monsters.yaml``.

Re-exports the Pydantic :class:`MonsterDefinition` as :class:`Monster`
so the engine's call sites don't have to spell ``MonsterDefinition``
everywhere.
"""

from __future__ import annotations

from pathlib import Path

import yaml

from app.game.schemas.monster_def import MonsterDefinition, MonstersFile

Monster = MonsterDefinition

_DEFAULT_PATH = Path(__file__).resolve().parents[2] / "data" / "bfrpg" / "monsters.yaml"

_monsters_cache: dict[str, MonsterDefinition] = {}
_loaded_path: Path | None = None


def _load_from(path: Path) -> dict[str, MonsterDefinition]:
    """Parse ``path`` into the validated mapping by monster name."""

    with path.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    if not isinstance(raw, list):
        raise ValueError(f"{path}: expected a top-level list of monsters")
    parsed = MonstersFile.model_validate({"monsters": raw})
    return {definition.name: definition for definition in parsed.monsters}


def load_monsters(path: Path | None = None) -> dict[str, MonsterDefinition]:
    """Return the cached monster table, loading on first call."""

    global _loaded_path
    if not _monsters_cache:
        target = path if path is not None else _DEFAULT_PATH
        _monsters_cache.update(_load_from(target))
        _loaded_path = target
    return _monsters_cache


def get_monster(name: str, *, path: Path | None = None) -> MonsterDefinition:
    """Look up a monster by name."""

    table = load_monsters(path)
    try:
        return table[name]
    except KeyError as exc:
        raise KeyError(f"unknown monster: {name!r}") from exc


def parse_hit_dice(notation: str) -> tuple[int, int]:
    """Decompose BFRPG hit-dice notation into ``(whole_dice, modifier)``.

    Examples::

        "1"        -> (1, 0)
        "1+1"      -> (1, 1)
        "1-1"      -> (1, -1)        # 1d8-1, BFRPG fractional
        "½" / "1/2"-> (0, 0)         # ½ HD: roll d4 in practice
        "3+2"      -> (3, 2)

    The returned ``whole_dice`` is the count of full d8s; ``modifier``
    is the additive constant. Fractional HD (``½``, ``1/2``) returns
    ``(0, 0)`` and the caller substitutes a ``d4``.
    """

    text = notation.strip()
    if text in {"½", "1/2"}:
        return 0, 0

    if "+" in text:
        base, mod = text.split("+", 1)
        return int(base), int(mod)
    if "-" in text:
        base, mod = text.split("-", 1)
        return int(base), -int(mod)
    return int(text), 0


def reset_cache() -> None:
    """Discard the in-memory monster table."""

    global _loaded_path
    _monsters_cache.clear()
    _loaded_path = None
