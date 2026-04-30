"""BFRPG rules engine — pure functions over plain dataclasses.

This module is the authoritative adjudicator for every mechanical
outcome. The LLM never resolves a die; it asks the engine via tool
calls and narrates whatever comes back. See spec §6 and the
``rules-engine`` agent description for the full contract.

Hard rules:

- Functions are pure: they read from the inputs, never mutate them,
  never touch I/O, never read clock or random state without an
  injected ``rng``. The call site persists the result to the database.
- Randomness is always injected as ``rng: random.Random``. Module-level
  :mod:`random` calls are forbidden.
- Engine returns dataclasses, never strings. Narration is the LLM's
  job.

The :class:`CharacterStats` dataclass is the engine's view of a
character. The persistence layer (``app.db.models.Character``)
converts to this on the way in. Keeping a wall between ORM rows and
engine inputs makes the engine usable in tests and from the chargen
code path without dragging SQLAlchemy in.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Literal

from app.game.dice import Roll, roll
from app.game.monsters import Monster

SaveKind = Literal["death_ray", "magic_wand", "paralysis", "dragon_breath", "spells"]
Ability = Literal["str", "int", "wis", "dex", "con", "cha"]


# --- Inputs the engine accepts -------------------------------------------------


@dataclass(frozen=True, slots=True)
class CharacterStats:
    """Engine-side view of a PC or NPC.

    ``attack_bonus`` is class+level derived (read from the class table
    by the call site before invoking the engine, so the engine doesn't
    need to know about loaders). ``saves`` is the resolved row from the
    class-saves table for this character's level — also resolved at the
    call site.

    ``save_modifiers`` is additive, typically supplied by the race
    (e.g. dwarf +4 on most saves). Applied as ``d20 + modifier ≥
    target``.
    """

    name: str
    class_name: str
    level: int
    hp_current: int
    hp_max: int
    ac: int
    str_score: int
    int_score: int
    wis_score: int
    dex_score: int
    con_score: int
    cha_score: int
    attack_bonus: int = 0
    saves: dict[SaveKind, int] = field(default_factory=dict)
    save_modifiers: dict[SaveKind, int] = field(default_factory=dict)
    hit_die: str = "d8"  # for level-up rolls; "d4".."d12"


# --- Result dataclasses --------------------------------------------------------


@dataclass(frozen=True, slots=True)
class AttackResult:
    """Outcome of a single attack roll.

    ``hit`` is the resolved success bool after natural-1/20 overrides.
    ``damage`` is rolled only when ``hit`` is true; otherwise zero.
    ``natural_one`` and ``natural_twenty`` mirror :class:`Roll` flags.
    ``attack_roll`` and ``damage_roll`` carry the underlying
    :class:`Roll`s for the audit log.
    """

    attacker: str
    target_ac: int
    weapon_name: str
    attack_roll: Roll
    hit: bool
    natural_one: bool
    natural_twenty: bool
    damage: int
    damage_roll: Roll | None


@dataclass(frozen=True, slots=True)
class SaveResult:
    """Outcome of a saving throw."""

    character: str
    save_kind: SaveKind
    target: int
    modifier: int
    roll: Roll
    success: bool


@dataclass(frozen=True, slots=True)
class CheckResult:
    """Outcome of an ability check (roll-under or DC-mode)."""

    character: str
    ability: Ability
    score: int
    modifier: int
    dc: int | None
    roll: Roll
    success: bool
    mode: Literal["roll_under", "dc"]


@dataclass(frozen=True, slots=True)
class DamageResult:
    """Pure damage computation: returns the new HP, no mutation.

    ``new_hp`` may be negative — call sites use ``new_hp <= 0`` as the
    trigger for the death-and-dismemberment table. ``below_zero_by``
    is the magnitude of the overrun (``0`` if still positive).
    """

    target: str
    amount: int
    source: str | None
    previous_hp: int
    new_hp: int
    dropped_to_zero: bool
    below_zero_by: int


@dataclass(frozen=True, slots=True)
class HealResult:
    """Pure heal computation. Cannot exceed ``hp_max``."""

    target: str
    amount: int
    previous_hp: int
    new_hp: int


@dataclass(frozen=True, slots=True)
class InitiativeOrder:
    """Resolved turn order for an encounter.

    ``entries`` is sorted descending by initiative roll, with ties
    broken deterministically by the second element of the entry — the
    participant id (or name). The ``round_number`` and ``index`` start
    at 1 and 0 respectively.
    """

    entries: list[InitiativeEntry]
    round_number: int = 1
    index: int = 0


@dataclass(frozen=True, slots=True)
class InitiativeEntry:
    participant_id: str
    name: str
    initiative: int
    roll: Roll
    is_player: bool


# --- Ability modifier ---------------------------------------------------------

# BFRPG curve, mapped 3..18 → modifier. Outside the canonical 3..18 range
# the curve extrapolates linearly: every two scores beyond 18 add another
# +1, every two below 3 subtract another -1. (Race ability minima/maxima
# normally keep PCs inside 3..18.)
_BFRPG_MODIFIER_TABLE: dict[int, int] = {
    3: -3,
    4: -2,
    5: -2,
    6: -1,
    7: -1,
    8: -1,
    9: 0,
    10: 0,
    11: 0,
    12: 0,
    13: 1,
    14: 1,
    15: 1,
    16: 2,
    17: 2,
    18: 3,
}


def ability_modifier(score: int) -> int:
    """Map a BFRPG ability score to its ``±N`` modifier.

    Inside ``3..18`` the table is canonical (3 → -3, 18 → +3). Outside
    that range the curve is extended linearly so artefact-buffed
    creatures still produce sensible output: 19/20 → +4, 21/22 → +5,
    etc.
    """

    if score in _BFRPG_MODIFIER_TABLE:
        return _BFRPG_MODIFIER_TABLE[score]
    if score < 3:
        # mirror the curve: 1-2 -> -4, -1..0 -> -5, etc.
        return -3 + (score - 3) // 2
    # score > 18: 19,20 → +4; 21,22 → +5; ...
    return 3 + (score - 17) // 2


# --- Attack roll --------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class WeaponSpec:
    """Minimal weapon view the attack resolver needs.

    The full :class:`app.game.items.Weapon` model has more fields; the
    call site passes whatever subset it has, or constructs this directly.
    """

    name: str
    damage: str  # dice expression, e.g. "1d8"
    is_ranged: bool = False


def attack_roll(
    attacker: CharacterStats,
    target_ac: int,
    weapon: WeaponSpec,
    *,
    rng: random.Random,
) -> AttackResult:
    """Resolve one attack: ``d20 + attack_bonus + str_mod ≥ target_ac``.

    Ranged weapons substitute Dex for Str on the to-hit modifier
    (BFRPG default; missile combat). Damage uses Str on melee, no
    ability bonus on ranged.

    Natural 20 always hits regardless of AC; natural 1 always misses
    regardless of bonuses. On a hit, damage is rolled from
    ``weapon.damage`` plus the appropriate ability modifier (Str for
    melee, none for ranged). Damage cannot drop below 1 on a successful
    hit.
    """

    ability_mod = (
        ability_modifier(attacker.dex_score)
        if weapon.is_ranged
        else ability_modifier(attacker.str_score)
    )
    to_hit_total_bonus = attacker.attack_bonus + ability_mod
    sign = "+" if to_hit_total_bonus >= 0 else "-"
    expression = f"1d20{sign}{abs(to_hit_total_bonus)}" if to_hit_total_bonus else "1d20"
    atk = roll(expression, rng=rng)

    if atk.natural_one:
        hit = False
    elif atk.natural_twenty:
        hit = True
    else:
        hit = atk.total >= target_ac

    if hit:
        # Damage: rolled die + (str_mod for melee). Ranged in core BFRPG
        # adds nothing; the spec has no Dex-to-damage variant.
        damage_bonus = ability_modifier(attacker.str_score) if not weapon.is_ranged else 0
        dmg_expr = weapon.damage
        if damage_bonus:
            sign = "+" if damage_bonus >= 0 else "-"
            dmg_expr = f"{weapon.damage}{sign}{abs(damage_bonus)}"
        dmg = roll(dmg_expr, rng=rng)
        damage_value = max(1, dmg.total)
        damage_roll: Roll | None = dmg
    else:
        damage_value = 0
        damage_roll = None

    return AttackResult(
        attacker=attacker.name,
        target_ac=target_ac,
        weapon_name=weapon.name,
        attack_roll=atk,
        hit=hit,
        natural_one=atk.natural_one,
        natural_twenty=atk.natural_twenty,
        damage=damage_value,
        damage_roll=damage_roll,
    )


# --- Saving throws ------------------------------------------------------------


def saving_throw(
    character: CharacterStats,
    save_kind: SaveKind,
    target: int | None = None,
    *,
    rng: random.Random,
) -> SaveResult:
    """Resolve a save: ``d20 + modifier ≥ target``.

    If ``target`` is omitted, the character's class-and-level save
    target for ``save_kind`` is used (read from
    ``character.saves[save_kind]``). The race's
    ``save_modifiers[save_kind]`` is added to the d20.
    """

    if target is None:
        try:
            target = character.saves[save_kind]
        except KeyError as exc:
            raise ValueError(
                f"no save target for {save_kind!r} on {character.name}; "
                "supply ``target`` or populate ``character.saves``"
            ) from exc
    modifier = character.save_modifiers.get(save_kind, 0)
    sign = "+" if modifier >= 0 else "-"
    expression = f"1d20{sign}{abs(modifier)}" if modifier else "1d20"
    save_roll = roll(expression, rng=rng)
    success = save_roll.total >= target
    return SaveResult(
        character=character.name,
        save_kind=save_kind,
        target=target,
        modifier=modifier,
        roll=save_roll,
        success=success,
    )


# --- Ability check ------------------------------------------------------------


def ability_check(
    character: CharacterStats,
    ability: Ability,
    dc: int | None = None,
    *,
    rng: random.Random,
) -> CheckResult:
    """Roll-under by default; DC-mode when ``dc`` is given.

    Roll-under: ``d20 ≤ score`` succeeds. A natural 20 always fails (a
    fumble convention; the engine is allowed to rule that), a natural 1
    always succeeds.

    DC-mode: ``d20 + modifier ≥ dc`` succeeds.
    """

    score = _ability_score(character, ability)
    modifier = ability_modifier(score)

    if dc is None:
        check_roll = roll("1d20", rng=rng)
        if check_roll.natural_one:
            success = True
        elif check_roll.natural_twenty:
            success = False
        else:
            success = check_roll.total <= score
        return CheckResult(
            character=character.name,
            ability=ability,
            score=score,
            modifier=modifier,
            dc=None,
            roll=check_roll,
            success=success,
            mode="roll_under",
        )

    sign = "+" if modifier >= 0 else "-"
    expression = f"1d20{sign}{abs(modifier)}" if modifier else "1d20"
    check_roll = roll(expression, rng=rng)
    success = check_roll.total >= dc
    return CheckResult(
        character=character.name,
        ability=ability,
        score=score,
        modifier=modifier,
        dc=dc,
        roll=check_roll,
        success=success,
        mode="dc",
    )


def _ability_score(character: CharacterStats, ability: Ability) -> int:
    """Get the named score off ``character`` (avoids ``getattr`` strings)."""

    return {
        "str": character.str_score,
        "int": character.int_score,
        "wis": character.wis_score,
        "dex": character.dex_score,
        "con": character.con_score,
        "cha": character.cha_score,
    }[ability]


# --- Damage / heal (pure) -----------------------------------------------------


def apply_damage(
    character: CharacterStats,
    amount: int,
    source: str | None = None,
) -> DamageResult:
    """Pure HP arithmetic. Returns the new HP — caller persists.

    Negative ``amount`` is rejected with :class:`ValueError`; healing
    has its own function.
    """

    if amount < 0:
        raise ValueError(f"damage amount must be non-negative; got {amount}")
    new_hp = character.hp_current - amount
    return DamageResult(
        target=character.name,
        amount=amount,
        source=source,
        previous_hp=character.hp_current,
        new_hp=new_hp,
        dropped_to_zero=new_hp <= 0,
        below_zero_by=max(0, -new_hp),
    )


def heal(character: CharacterStats, amount: int) -> HealResult:
    """Pure heal: cannot exceed ``hp_max``, cannot revive a 0-HP character.

    The 0-HP case is left to the death-and-dismemberment table /
    explicit revival rules; passing a 0-HP character through here is a
    programming error and raises :class:`ValueError`.
    """

    if amount < 0:
        raise ValueError(f"heal amount must be non-negative; got {amount}")
    if character.hp_current <= 0:
        raise ValueError(
            f"{character.name} is at {character.hp_current} HP — heal via revival, "
            "not by ordinary heal()"
        )
    new_hp = min(character.hp_max, character.hp_current + amount)
    return HealResult(
        target=character.name,
        amount=amount,
        previous_hp=character.hp_current,
        new_hp=new_hp,
    )


# --- Level-up HP --------------------------------------------------------------


def hp_at_level_up(character: CharacterStats, *, rng: random.Random) -> int:
    """Roll the class hit die plus Con modifier; minimum 1 gained."""

    die_roll = roll(f"1{character.hit_die}", rng=rng)
    con_mod = ability_modifier(character.con_score)
    return max(1, die_roll.total + con_mod)


# --- XP -----------------------------------------------------------------------


def xp_for_treasure(gp_value: int) -> int:
    """House rule (spec §4): 1 XP per 1 gp recovered."""

    if gp_value < 0:
        raise ValueError(f"gp_value must be non-negative; got {gp_value}")
    return gp_value


def encounter_xp(monsters: list[Monster]) -> int:
    """Sum the per-monster XP awards. Empty list → 0."""

    return sum(m.xp for m in monsters)


# --- Initiative ---------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Participant:
    """Anything that takes a turn in initiative order — PC or monster.

    ``dex_modifier`` is the dexterity-based modifier (BFRPG core uses
    individual initiative as an optional rule; we apply Dex mod by
    default since the rest of the engine has Dex available). Pass 0
    for monsters without a dex score.
    """

    participant_id: str
    name: str
    dex_modifier: int = 0
    is_player: bool = False


def roll_initiative(
    participants: list[Participant],
    *,
    rng: random.Random,
) -> InitiativeOrder:
    """Roll 1d6 + Dex mod for each participant; sort descending.

    Ties are broken deterministically by ``participant_id`` (lower
    sorts first). The output is reproducible given the same ``rng``
    seed and input list ordering.
    """

    entries: list[InitiativeEntry] = []
    for p in participants:
        sign = "+" if p.dex_modifier >= 0 else "-"
        expr = f"1d6{sign}{abs(p.dex_modifier)}" if p.dex_modifier else "1d6"
        r = roll(expr, rng=rng)
        entries.append(
            InitiativeEntry(
                participant_id=p.participant_id,
                name=p.name,
                initiative=r.total,
                roll=r,
                is_player=p.is_player,
            )
        )
    # Sort descending by initiative; ties broken by participant_id ascending.
    entries.sort(key=lambda e: (-e.initiative, e.participant_id))
    return InitiativeOrder(entries=entries, round_number=1, index=0)
