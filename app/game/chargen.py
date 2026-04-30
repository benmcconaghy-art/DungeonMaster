"""Character generation.

Two ability-score methods are supported:

- ``classic`` (default, BFRPG-canonical): ``3d6`` per ability in order
  STR / INT / WIS / DEX / CON / CHA.
- ``heroic``: ``4d6`` drop the lowest, per ability, in order. (Common
  house rule for less-lethal campaigns; not to be confused with the
  point-buy or array methods, which we don't implement.)

The output of :func:`generate_character` is a :class:`GeneratedCharacter`
— the full level-1 stat block ready to be persisted as a
``characters`` row, plus a starting gold figure (``3d6 * 10`` per
BFRPG core).

The function validates race+class compatibility against the loaded
race table; an invalid pair raises :class:`ValueError`. Ability-score
requirements (e.g. Dwarf STR ≥ 9) are *not* re-rolled — if the rolled
scores fail the requirement the function raises and the caller can
re-roll. (That keeps the function pure and deterministic given a seed.)
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Literal

from app.game.classes import get_class
from app.game.dice import roll
from app.game.races import get_race, race_allows_class
from app.game.rules import Ability, CharacterStats, SaveKind, ability_modifier
from app.game.schemas.race_def import RaceDefinition

AbilityMethod = Literal["classic", "heroic"]

ABILITY_ORDER: tuple[Ability, ...] = ("str", "int", "wis", "dex", "con", "cha")


@dataclass(frozen=True, slots=True)
class AbilityScores:
    """3d6-in-order (or 4d6-drop-lowest) result.

    Attributes mirror :class:`CharacterStats` ``*_score`` fields.
    """

    str_score: int
    int_score: int
    wis_score: int
    dex_score: int
    con_score: int
    cha_score: int

    def get(self, ability: Ability) -> int:
        return {
            "str": self.str_score,
            "int": self.int_score,
            "wis": self.wis_score,
            "dex": self.dex_score,
            "con": self.con_score,
            "cha": self.cha_score,
        }[ability]


@dataclass(frozen=True, slots=True)
class GeneratedCharacter:
    """The full level-1 result of :func:`generate_character`.

    The call site converts this to a ``characters`` ORM row. ``stats``
    is the engine's :class:`CharacterStats` view used for any further
    in-engine resolution before persistence.
    """

    name: str
    race: str
    class_name: str
    alignment: str
    level: int
    abilities: AbilityScores
    hp_max: int
    ac: int
    saves: dict[SaveKind, int]
    save_modifiers: dict[SaveKind, int]
    starting_gold: int
    stats: CharacterStats


def roll_abilities(method: AbilityMethod = "classic", *, rng: random.Random) -> AbilityScores:
    """Roll a fresh ability-score block.

    ``classic`` rolls ``3d6`` per ability in order. ``heroic`` rolls
    ``4d6`` and drops the lowest, again in order.
    """

    expr = "3d6" if method == "classic" else "4d6kh3"
    rolls = [roll(expr, rng=rng).total for _ in ABILITY_ORDER]
    return AbilityScores(
        str_score=rolls[0],
        int_score=rolls[1],
        wis_score=rolls[2],
        dex_score=rolls[3],
        con_score=rolls[4],
        cha_score=rolls[5],
    )


def _check_race_requirements(scores: AbilityScores, race: RaceDefinition) -> None:
    """Verify ``scores`` meet ``race.ability_requirements``.

    Raises :class:`ValueError` with a clear summary if any ability fails
    its min/max gate. Caller is expected to re-roll or pick another
    race.
    """

    failures: list[str] = []
    for ability, req in race.ability_requirements.items():
        score = scores.get(ability)
        if req.min is not None and score < req.min:
            failures.append(f"{ability.upper()} {score} < min {req.min}")
        if req.max is not None and score > req.max:
            failures.append(f"{ability.upper()} {score} > max {req.max}")
    if failures:
        raise ValueError(f"ability scores fail {race.name} requirements: " + ", ".join(failures))


def starting_hp(class_hit_die: str, con_mod: int) -> int:
    """Level-1 HP: max die + Con modifier, minimum 1.

    BFRPG convention is that level 1 takes the die's maximum rather
    than rolling — keeps starting characters from being one-shotted by
    a goblin's lucky scimitar.
    """

    die_size = int(class_hit_die.lstrip("d"))
    return max(1, die_size + con_mod)


def starting_ac(dex_mod: int, base_ac: int = 11) -> int:
    """Default starting AC: ``11 + dex_mod`` (no armour).

    Equipment-aware AC is computed at the call site once the character
    is outfitted from ``equipment.yaml``.
    """

    return base_ac + dex_mod


def starting_gold(*, rng: random.Random) -> int:
    """``3d6 * 10`` gp per BFRPG core."""

    return roll("3d6", rng=rng).total * 10


def generate_character(
    *,
    name: str,
    race_name: str,
    class_name: str,
    alignment: str,
    rng: random.Random,
    method: AbilityMethod = "classic",
    abilities: AbilityScores | None = None,
) -> GeneratedCharacter:
    """Roll a full level-1 character.

    Pass ``abilities`` to skip the roll (useful for testing fixed
    statblocks). Otherwise the function rolls per ``method``.

    The function validates:

    - Race exists in the loaded ``races.yaml``.
    - Class exists in the loaded ``classes.yaml``.
    - The race allows the chosen class.
    - The rolled scores meet the race's ability requirements.
    - The chosen alignment is in the race's allowed list.
    """

    race = get_race(race_name)
    class_def = get_class(class_name)
    if not race_allows_class(race, class_name):
        raise ValueError(
            f"race {race_name!r} cannot take class {class_name!r}; "
            f"allowed: {race.allowed_classes}"
        )
    if alignment not in race.allowed_alignments:
        raise ValueError(
            f"alignment {alignment!r} not allowed for {race_name}; "
            f"allowed: {race.allowed_alignments}"
        )

    scores = abilities if abilities is not None else roll_abilities(method, rng=rng)
    _check_race_requirements(scores, race)

    con_mod = ability_modifier(scores.con_score)
    dex_mod = ability_modifier(scores.dex_score)

    # ``model_dump`` returns a generic ``dict[str, Any]``; the schema
    # constrains the keys to exactly the five SaveKind literals so we
    # build the typed mapping explicitly to keep mypy strict-happy.
    save_row = class_def.saves[1]
    typed_saves: dict[SaveKind, int] = {
        "death_ray": save_row.death_ray,
        "magic_wand": save_row.magic_wand,
        "paralysis": save_row.paralysis,
        "dragon_breath": save_row.dragon_breath,
        "spells": save_row.spells,
    }
    save_modifiers: dict[SaveKind, int] = dict(race.save_modifiers)

    hp_max = starting_hp(class_def.hit_die, con_mod)
    ac = starting_ac(dex_mod)
    gold = starting_gold(rng=rng)

    stats = CharacterStats(
        name=name,
        class_name=class_name,
        level=1,
        hp_current=hp_max,
        hp_max=hp_max,
        ac=ac,
        str_score=scores.str_score,
        int_score=scores.int_score,
        wis_score=scores.wis_score,
        dex_score=scores.dex_score,
        con_score=scores.con_score,
        cha_score=scores.cha_score,
        attack_bonus=class_def.attack_bonus_progression[1],
        saves=typed_saves,
        save_modifiers=save_modifiers,
        hit_die=class_def.hit_die,
    )

    return GeneratedCharacter(
        name=name,
        race=race_name,
        class_name=class_name,
        alignment=alignment,
        level=1,
        abilities=scores,
        hp_max=hp_max,
        ac=ac,
        saves=typed_saves,
        save_modifiers=save_modifiers,
        starting_gold=gold,
        stats=stats,
    )
