"""BFRPG rules engine.

The engine is the authoritative adjudicator for every mechanical
outcome (spec §6). Public surface lives in :mod:`app.game.rules`,
:mod:`app.game.dice`, :mod:`app.game.combat`, :mod:`app.game.death`,
and :mod:`app.game.chargen`. Content loaders for the YAML files under
``data/bfrpg/`` are :mod:`app.game.classes`, :mod:`app.game.races`,
:mod:`app.game.items`, and :mod:`app.game.monsters`. The validator
script (:mod:`app.game.validate_data`) is the CI gate for those files.
"""

from __future__ import annotations
