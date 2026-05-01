"""Dice expression parser and roller.

The single entry point is :func:`roll`, which evaluates an expression like
``1d20+5`` or ``4d6kh3`` against an injected :class:`random.Random` and
returns a :class:`Roll` with the total, the individual rolled values, and
flags for natural-1 / natural-20 (useful at the call site for crit /
fumble logic — kept as advisory data here; the engine's combat resolver
decides what to do with them).

Grammar (case-insensitive)::

    expr     := "(" dice (keep)? ")" (mult)? (mod)?
              | dice (keep)? (mult)? (mod)?
    dice     := N "d" M
    keep     := ("kh" | "kl") K
    mult     := "*" K
    mod      := ("+" | "-") K

Where ``N``, ``M``, ``K`` are positive integers. ``N`` defaults to ``1``
(``d20`` parses the same as ``1d20``). The multiplier is the BFRPG idiom
for things like starting gold (``3d6*10``); the math is
``(sum_of_kept * mult) + modifier`` with mult defaulting to ``1`` and
modifier to ``0``. Whitespace around ``*`` / ``+`` / ``-`` is tolerated.
A single layer of parentheses around the dice term is allowed
(``(3d6)*10``); deeper nesting or a full expression parser is out of
scope — this module only does dice + scalar arithmetic.

Advantage / disadvantage are passed as keyword arguments. The full
expression is rolled twice and the better (advantage) or worse
(disadvantage) total is selected. The losing roll is discarded — only
the kept values appear on the returned :class:`Roll`. Advantage is
mostly meaningful for d20s, but the parameter is kept generic; callers
get to decide.

No module-level :mod:`random` use, ever — the caller injects an
``rng: random.Random`` so tests are deterministic.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class Roll:
    """The structured result of evaluating a dice expression.

    ``individual`` is every die value that contributed to ``total``,
    after keep-highest / keep-lowest filtering. The dropped dice are
    discarded — if a caller needs to display them, parse them
    themselves. ``natural_one`` / ``natural_twenty`` are convenience
    flags that fire only on a single-d20 expression (the most common
    case for crits/fumbles); for ``4d6kh3`` they're always ``False``.
    """

    expression: str
    total: int
    individual: list[int] = field(default_factory=list)
    natural_one: bool = False
    natural_twenty: bool = False


# Anchored, case-insensitive. The dice term may be wrapped in a single
# layer of parens to support the ``(3d6)*10`` idiom; the parser balances
# the parens itself rather than going through a real expression parser.
# Groups: count, faces, keep-mode, keep-count, mult, sign, modifier.
_DICE_RE = re.compile(
    r"""
    ^\s*
    (?P<lparen>\()?         # optional opening paren around the dice term
    \s*
    (?P<count>\d+)?         # optional die count, defaults to 1
    d
    (?P<faces>\d+)
    (?:                     # optional keep-highest / keep-lowest
      k(?P<keep_mode>[hl])
      (?P<keep_count>\d+)
    )?
    \s*
    (?P<rparen>\))?         # optional closing paren around the dice term
    (?:                     # optional scalar multiplier (BFRPG idiom: 3d6*10)
      \s*\*\s*
      (?P<mult>\d+)
    )?
    (?:                     # optional flat modifier
      \s*(?P<sign>[+-])\s*
      (?P<modifier>\d+)
    )?
    \s*$
    """,
    re.IGNORECASE | re.VERBOSE,
)


@dataclass(frozen=True, slots=True)
class _Parsed:
    count: int
    faces: int
    keep_mode: str | None  # 'h', 'l', or None
    keep_count: int | None
    multiplier: int  # defaults to 1
    modifier: int  # signed; already incorporates +/-


def _parse(expression: str) -> _Parsed:
    """Parse ``expression`` into its components, or raise ``ValueError``."""

    match = _DICE_RE.match(expression)
    if match is None:
        raise ValueError(f"invalid dice expression: {expression!r}")

    # Parens must match: either both present or both absent. A lone
    # opening or closing paren is malformed.
    lparen = match.group("lparen")
    rparen = match.group("rparen")
    if bool(lparen) != bool(rparen):
        raise ValueError(f"unbalanced parentheses in dice expression: {expression!r}")

    count_str = match.group("count")
    count = int(count_str) if count_str is not None else 1
    faces = int(match.group("faces"))
    if count <= 0 or faces <= 0:
        raise ValueError(f"dice count and faces must be positive: {expression!r}")

    keep_mode_raw = match.group("keep_mode")
    keep_mode = keep_mode_raw.lower() if keep_mode_raw is not None else None
    keep_count_str = match.group("keep_count")
    keep_count = int(keep_count_str) if keep_count_str is not None else None
    if keep_mode is not None:
        # parser guarantees keep_count is set whenever keep_mode is set
        assert keep_count is not None
        if keep_count <= 0 or keep_count > count:
            raise ValueError(
                f"keep count must be in 1..{count} for {expression!r}; got {keep_count}"
            )

    mult_str = match.group("mult")
    multiplier = 1
    if mult_str is not None:
        multiplier = int(mult_str)
        if multiplier <= 0:
            raise ValueError(f"dice multiplier must be positive: {expression!r}")

    sign = match.group("sign")
    modifier_str = match.group("modifier")
    modifier = 0
    if modifier_str is not None:
        modifier = int(modifier_str)
        if sign == "-":
            modifier = -modifier

    return _Parsed(
        count=count,
        faces=faces,
        keep_mode=keep_mode,
        keep_count=keep_count,
        multiplier=multiplier,
        modifier=modifier,
    )


def _roll_once(parsed: _Parsed, rng: random.Random) -> tuple[int, list[int]]:
    """One evaluation of a parsed expression. Returns ``(total, kept)``."""

    rolls = [rng.randint(1, parsed.faces) for _ in range(parsed.count)]

    if parsed.keep_mode is None:
        kept = list(rolls)
    elif parsed.keep_mode == "h":
        assert parsed.keep_count is not None
        kept = sorted(rolls, reverse=True)[: parsed.keep_count]
    else:  # 'l'
        assert parsed.keep_count is not None
        kept = sorted(rolls)[: parsed.keep_count]

    total = sum(kept) * parsed.multiplier + parsed.modifier
    return total, kept


def roll(
    expression: str,
    *,
    advantage: bool = False,
    disadvantage: bool = False,
    rng: random.Random,
) -> Roll:
    """Evaluate ``expression`` and return the structured :class:`Roll`.

    ``advantage`` and ``disadvantage`` are mutually exclusive — passing
    both raises :class:`ValueError`. With either set, the entire
    expression is rolled twice and the higher (advantage) or lower
    (disadvantage) total is kept. The losing roll is discarded.
    """

    if advantage and disadvantage:
        raise ValueError("advantage and disadvantage are mutually exclusive")

    parsed = _parse(expression)

    total, individual = _roll_once(parsed, rng)
    if advantage or disadvantage:
        alt_total, alt_individual = _roll_once(parsed, rng)
        pick_alt = (advantage and alt_total > total) or (disadvantage and alt_total < total)
        if pick_alt:
            total, individual = alt_total, alt_individual

    # Natural 1 / 20 only meaningful for a single d20 with no keep filter and no modifier
    # contribution to the "natural" status. Convention: the *die* showed 1 or 20.
    nat_one = False
    nat_twenty = False
    if parsed.count == 1 and parsed.faces == 20 and parsed.keep_mode is None:
        # For advantage/disadvantage we only inspect the kept roll.
        die = individual[0]
        nat_one = die == 1
        nat_twenty = die == 20

    return Roll(
        expression=expression,
        total=total,
        individual=individual,
        natural_one=nat_one,
        natural_twenty=nat_twenty,
    )
