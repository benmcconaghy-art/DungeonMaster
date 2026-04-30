"""Condensed BFRPG rules text for the DM system prompt.

Roughly 1500 tokens covering AC, saves, classes, spell-casting, the combat
sequence, and conditions. Injected into every prompt so the DM adjudicates
consistently within the BFRPG ruleset (spec §7).

The text is *generated*, not hand-rolled. Class hit dice, attack
progression archetypes, and other tabular facts are pulled from the
canonical YAML in ``data/bfrpg/classes.yaml`` so the prompt stays in sync
with the rules engine. The ability-modifier curve comes from the rules
engine itself (:func:`app.game.rules.ability_modifier`) — same source of
truth the engine uses to adjudicate.

Sourced from the BFRPG core rulebook (CC BY-SA). Cached in module state
so re-rendering on every turn is free.
"""

from __future__ import annotations

from app.game.classes import load_classes
from app.game.rules import ability_modifier

_cached_text: str | None = None


def _ability_modifier_table() -> str:
    """Render the BFRPG ability-modifier curve (3..18) as one line."""

    pairs = [f"{score}={ability_modifier(score):+d}" for score in range(3, 19)]
    return "  Modifier table: " + ", ".join(pairs) + "."


def _saves_summary(class_name: str) -> str:
    """One-line saves snapshot at L1 for a class. Grim but accurate."""

    classes = load_classes()
    cls = classes[class_name]
    saves = cls.saves[1]
    return (
        f"{class_name} (HD {cls.hit_die}): "
        f"DR {saves.death_ray}, MW {saves.magic_wand}, "
        f"P {saves.paralysis}, DB {saves.dragon_breath}, S {saves.spells}"
    )


def _attack_progression_summary(class_name: str) -> str:
    """One-line attack-bonus progression snapshot for a class."""

    classes = load_classes()
    cls = classes[class_name]
    progression = cls.attack_bonus_progression
    levels = sorted(progression.keys())
    sample = [(lvl, progression[lvl]) for lvl in levels if lvl in (1, 4, 8, 12)]
    return f"{class_name} attack bonus: " + ", ".join(f"L{lvl}=+{ab}" for lvl, ab in sample)


def render_rules_text() -> str:
    """Return the condensed BFRPG rules block for the DM prompt.

    The rendered text is cached; subsequent calls return the same string.
    Tests that need to re-render after monkey-patching the YAML loader
    should call :func:`reset_cache` first.
    """

    global _cached_text
    if _cached_text is not None:
        return _cached_text

    classes = load_classes()
    class_names = sorted(classes)

    lines: list[str] = []

    lines.append("=== BFRPG core mechanics ===")
    lines.append("")
    lines.append("CORE RESOLUTION")
    lines.append("  Attacks: d20 + attack_bonus + ability_mod >= target AC. Ascending AC only.")
    lines.append("  Natural 20 always hits; natural 1 always misses.")
    lines.append(
        "  Saving throws: d20 + (race/situational mods) >= target N. Lower target = harder save."
    )
    lines.append(
        "  Ability checks (default): roll-under d20 <= ability score. Nat 1 succeeds, nat 20 fails."
    )
    lines.append("  Ability checks (DC mode, when DC supplied): d20 + ability_mod >= DC.")
    lines.append("")

    lines.append("ABILITY MODIFIERS (BFRPG curve)")
    lines.append(_ability_modifier_table())
    lines.append(
        "  Use STR for melee attack and damage; DEX for missile attack;"
        " CON modifies HP per level; INT/WIS/CHA for class abilities and reactions."
    )
    lines.append("")

    lines.append("SAVING THROWS — five categories (lower target = better)")
    lines.append("  Death Ray / Poison; Magic Wand; Paralysis / Petrify;")
    lines.append("  Dragon Breath; Spells / Rod / Staff.")
    lines.append("  Class+level determines targets. L1 sample:")
    for save_cls_name in class_names:
        lines.append("    " + _saves_summary(save_cls_name))
    lines.append("")

    lines.append("HP AND DAMAGE")
    lines.append("  HP per level: roll class hit die + CON mod (minimum 1 gained).")
    lines.append("  Damage reduces hp_current. At hp_current <= 0 the engine rolls on the")
    lines.append("  Death and Dismemberment table (house rule). Outcomes range from")
    lines.append("  knocked-out (1d6 turns unconscious, hp 0) to dead (instant kill).")
    lines.append("  Healing caps at hp_max. Cannot heal a 0-HP character via ordinary heal;")
    lines.append("  revival is its own ritual.")
    lines.append("")

    lines.append("COMBAT SEQUENCE (per round)")
    lines.append("  1. Initiative: 1d6 + DEX mod per side or per participant; high goes first.")
    lines.append("  2. On your turn: move (40' typical) and act (attack, spell, item, etc.).")
    lines.append("  3. Attacks: pick a target, call request_dice_roll for the to-hit, then")
    lines.append("     request_dice_roll for damage. The engine adjudicates hit/miss.")
    lines.append("  4. Spells: declare at start of round; concentration is broken by damage.")
    lines.append("  5. End of round: apply ongoing effects (bleeding, poison, etc.).")
    lines.append("")

    lines.append("CONDITIONS (BFRPG-light)")
    lines.append("  Prone: -4 to attacks, +4 to be hit in melee. Stand up uses half move.")
    lines.append("  Held / paralysed: no action; melee attackers hit automatically.")
    lines.append("  Unconscious (0 HP): no action; awake at hp 1 or after Death table outcome.")
    lines.append("  Frightened: -2 to attacks and saves. Flees toward safety on next turn.")
    lines.append("  Charmed: treats caster as trusted ally; will not attack caster directly.")
    lines.append("")

    lines.append("CLASS HIGHLIGHTS")
    for cls_name in class_names:
        cls = classes[cls_name]
        abilities = ", ".join(cls.special_abilities) if cls.special_abilities else "—"
        lines.append(f"  {cls_name} (HD {cls.hit_die}, prime: {cls.prime_requisite}): {abilities}")
    lines.append("  Magic-Users and Clerics gain spells per spec; Thief skills are percentile.")
    lines.append("")

    lines.append("ATTACK PROGRESSION ARCHETYPES")
    for cls_name in class_names:
        lines.append("  " + _attack_progression_summary(cls_name))
    lines.append("")

    lines.append("DM DISCIPLINE (engine adjudicates, you narrate)")
    lines.append("  - Never invent a die outcome. Always call request_dice_roll.")
    lines.append("  - Never declare HP changes in prose. Always call apply_damage / heal.")
    lines.append("  - Never declare a location change in prose. Always call transition_location.")
    lines.append("  - Never narrate initiative or monster HP totals. The engine owns those.")
    lines.append("  - When the engine returns a result, narrate that result faithfully.")
    lines.append("=== end BFRPG core mechanics ===")

    _cached_text = "\n".join(lines)
    return _cached_text


def reset_cache() -> None:
    """Discard the cached rendered text. Tests use this to re-render after
    swapping out the underlying YAML data."""

    global _cached_text
    _cached_text = None


__all__ = ["render_rules_text", "reset_cache"]
