"""Loader for ``data/bfrpg/classes.yaml``.

The validated Pydantic model (:class:`ClassDefinition`) is what we
expose to the rest of the engine — there's no second, lighter
dataclass layer. Pydantic v2 attribute access is fast enough that the
duplication isn't worth the maintenance cost.

The loader caches the parsed file in module state so test isolation
needs ``_classes_cache.clear()`` (exposed for that purpose). In
production the cache is a one-time cost at app startup.

If the YAML file is missing the loader raises :class:`FileNotFoundError`;
the validator script (``app.game.validate_data``) handles that
gracefully so a fresh checkout doesn't crash before the bfrpg-data
agent has filled in content.
"""

from __future__ import annotations

from pathlib import Path

import yaml

from app.game.schemas.class_def import ClassDefinition, ClassesFile

_DEFAULT_PATH = Path(__file__).resolve().parents[2] / "data" / "bfrpg" / "classes.yaml"

_classes_cache: dict[str, ClassDefinition] = {}
_loaded_path: Path | None = None


def _load_from(path: Path) -> dict[str, ClassDefinition]:
    """Parse ``path`` (YAML) into the validated mapping by class name."""

    with path.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    if not isinstance(raw, list):
        raise ValueError(f"{path}: expected a top-level list of classes")
    parsed = ClassesFile.model_validate({"classes": raw})
    return {definition.name: definition for definition in parsed.classes}


def load_classes(path: Path | None = None) -> dict[str, ClassDefinition]:
    """Return the cached class table, loading from ``path`` on first call.

    Subsequent calls reuse the cache regardless of ``path`` — pass
    ``path`` only on the first invocation, or call
    :func:`_classes_cache.clear` first to swap files (useful in tests).
    """

    global _loaded_path
    if not _classes_cache:
        target = path if path is not None else _DEFAULT_PATH
        _classes_cache.update(_load_from(target))
        _loaded_path = target
    return _classes_cache


def get_class(name: str, *, path: Path | None = None) -> ClassDefinition:
    """Look up a class by name. Raises :class:`KeyError` if unknown."""

    table = load_classes(path)
    try:
        return table[name]
    except KeyError as exc:
        raise KeyError(f"unknown class: {name!r} (known: {sorted(table)})") from exc


def reset_cache() -> None:
    """Discard the in-memory class table. Tests call this between cases."""

    global _loaded_path
    _classes_cache.clear()
    _loaded_path = None
