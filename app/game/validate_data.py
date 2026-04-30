"""Standalone validator for ``data/bfrpg/*.yaml``.

Runnable as ``python -m app.game.validate_data``. Loads each YAML
content file in turn and validates it against the matching Pydantic
schema. Prints a per-file summary; exits with status 1 on the first
failure so CI can gate on it.

If a file is missing the validator reports it as ``not yet authored``
and continues — the bfrpg-data agent fills in YAML files in parallel
with the engine work, so a fresh checkout normally has no content
files. The script returns 0 when every file present validates and at
least one validation has been attempted; it returns 0 also when no
files are present (with a clear "no data files yet" message).
"""

from __future__ import annotations

import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import yaml
from pydantic import BaseModel, ValidationError

from app.game.schemas.class_def import ClassesFile
from app.game.schemas.equipment_def import EquipmentFile
from app.game.schemas.monster_def import MonstersFile
from app.game.schemas.race_def import RacesFile
from app.game.schemas.spell_def import SpellsFile

DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "bfrpg"


@dataclass(frozen=True, slots=True)
class _Target:
    """One YAML file → schema mapping for the validator to walk."""

    filename: str
    schema: type[BaseModel]
    # For top-level lists, the schema wraps the list under a single key.
    # For top-level mappings (equipment), the schema validates as-is.
    wrap_key: str | None


_TARGETS: tuple[_Target, ...] = (
    _Target("classes.yaml", ClassesFile, "classes"),
    _Target("races.yaml", RacesFile, "races"),
    _Target("spells.yaml", SpellsFile, "spells"),
    _Target("monsters.yaml", MonstersFile, "monsters"),
    _Target("equipment.yaml", EquipmentFile, None),
)


def _validate_file(target: _Target, *, log: Callable[[str], None]) -> bool:
    """Return ``True`` on success, ``False`` on validation failure.

    A missing file is logged but not treated as a failure (the
    bfrpg-data agent may not have authored it yet). Anything else
    (parse error, schema violation) is a hard failure.
    """

    path = DATA_DIR / target.filename
    if not path.exists():
        log(f"  - {target.filename:18s} not yet authored — skipping")
        return True

    try:
        with path.open("r", encoding="utf-8") as fh:
            raw = yaml.safe_load(fh)
    except yaml.YAMLError as exc:
        log(f"  - {target.filename:18s} FAIL — YAML parse error: {exc}")
        return False

    if target.wrap_key is not None:
        if not isinstance(raw, list):
            log(
                f"  - {target.filename:18s} FAIL — expected top-level list, "
                f"got {type(raw).__name__}"
            )
            return False
        payload = {target.wrap_key: raw}
    else:
        if not isinstance(raw, dict):
            log(
                f"  - {target.filename:18s} FAIL — expected top-level mapping, "
                f"got {type(raw).__name__}"
            )
            return False
        payload = raw

    try:
        target.schema.model_validate(payload)
    except ValidationError as exc:
        log(f"  - {target.filename:18s} FAIL — {exc.error_count()} issue(s):")
        for err in exc.errors():
            loc = ".".join(str(p) for p in err["loc"])
            log(f"      {loc}: {err['msg']}")
        return False

    if target.wrap_key is not None and isinstance(raw, list):
        count = len(raw)
        log(f"  - {target.filename:18s} OK ({count} entries)")
    else:
        log(f"  - {target.filename:18s} OK")
    return True


def validate_all(*, log: Callable[[str], None] = print) -> int:
    """Validate every target. Return the process exit code (0 / 1)."""

    log(f"Validating BFRPG content in {DATA_DIR}")
    if not DATA_DIR.exists():
        log("  data directory does not exist — no data files yet")
        return 0

    present = [t for t in _TARGETS if (DATA_DIR / t.filename).exists()]
    if not present:
        log("  no data files yet — nothing to validate")
        return 0

    all_ok = True
    for target in _TARGETS:
        ok = _validate_file(target, log=log)
        all_ok = all_ok and ok

    log("")
    log("OK" if all_ok else "FAILED")
    return 0 if all_ok else 1


def main() -> int:
    """Entry point for ``python -m app.game.validate_data``."""

    return validate_all()


if __name__ == "__main__":
    sys.exit(main())
