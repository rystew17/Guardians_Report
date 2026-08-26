"""Where each market's probability comes from, and how sure we may be about it.

Beliefs are keyed by market, selection *and* line. The line is not decoration:
books post alternate numbers on the same bet, and Mike Trout's hits were quoted
at both 0.5 and 1.5 on the same night. Keyed without it, the second wrote over
the first and his chance of clearing 0.5 hits was reported as his chance of
clearing 1.5 -- 79% where the truth was 37%, displayed against the 0.5 line.


One market at a time, because they do not share a model and they do not share
an evidence base:

* **Moneyline** -- Model A, the calibrated logistic. The only market with a
  measured standard error behind it, so the only one that can currently be
  staked.
* **Total** -- Model B's simulated joint score distribution.
* **First five** -- the dedicated F5 fit.
* **Strikeouts** -- the starter's own distribution from the props model.

The last three are marked unmeasured. That is not a placeholder: no calibration
data exists for them, so there is no honest basis for a standard error, and
`guide.build` refuses to stake anything carrying that flag. They still appear on
the page, because a disagreement is worth seeing even when it cannot be sized.

Selections are keyed lower-case so a hand-typed `cle` matches `CLE`. Anything
that fails to match is surfaced by the guide rather than dropped, since a
mistyped team code and a night with no edge otherwise look identical.
"""

from __future__ import annotations

from typing import Any

from guards_report.betting import uncertainty
from guards_report.betting.guide import Belief
from guards_report.odds import types

# What a market's probability rests on, for the page to show beside it.
BASIS = {
    types.MONEYLINE: "Model A, calibrated logistic",
    types.TOTAL: "Model B, simulated score distribution",
    types.F5_MONEYLINE: "first-five model",
    types.F5_TOTAL: "first-five model",
    types.STRIKEOUTS: "starter strikeout distribution",
    types.HITS: "batter hit distribution",
    types.HOME_RUNS: "batter home run distribution",
    types.TOTAL_BASES: "batter total base distribution",
    types.RUNLINE: "Model B, simulated margin distribution",
}


def first_five_total(projection, line: float,
                     calibration: dict | None = None) -> dict[tuple[str, str], Belief]:
    """P(the first five innings go over the posted number).

    The first-five model carries expected runs per side rather than a joint
    distribution, so the total is drawn from two negative binomials with the
    same dispersion the model was fitted under. Same shape as the full-game
    total, over five innings instead of nine.
    """
    if projection is None:
        return {}
    home = float(getattr(projection, "expected_home", 0.0) or 0.0)
    away = float(getattr(projection, "expected_away", 0.0) or 0.0)
    if home <= 0 or away <= 0:
        return {}

    import numpy as np

    from guards_report.projections import first5 as f5_module

    alpha = f5_module.FIRST5_ALPHA
    n = 1.0 / alpha
    rng = np.random.default_rng(20260825)
    draws = 20_000
    total = (rng.negative_binomial(n, n / (n + home), draws)
             + rng.negative_binomial(n, n / (n + away), draws))

    over = float((total > line).mean())
    push = float((total == line).mean())
    live = 1.0 - push
    if live <= 0:
        return {}
    p_over = over / live

    sigma = uncertainty.market_sigma(
        calibration or {}, types.F5_TOTAL, p_over, line)
    return {
        (types.F5_TOTAL, "over", line): Belief(
            probability=p_over, sigma=sigma or uncertainty.MINIMUM_SIGMA,
            measured=sigma is not None, basis=BASIS[types.F5_TOTAL]),
        (types.F5_TOTAL, "under", line): Belief(
            probability=1.0 - p_over, sigma=sigma or uncertainty.MINIMUM_SIGMA,
            measured=sigma is not None, basis=BASIS[types.F5_TOTAL]),
    }


def moneyline(
    outcome_model,
    features: dict[str, float],
    *,
    home: str,
    away: str,
    home_aliases: tuple[str, ...] = (),
    away_aliases: tuple[str, ...] = (),
    market: str = types.MONEYLINE,
) -> dict[tuple[str, str], Belief]:
    """P(home win) and its complement, with the measured standard error.

    Registered under every name a side goes by. Hand-typed prices use the
    abbreviation ("CLE"); the odds feed uses the full club name ("Cleveland
    Guardians"). Keying on one of them means the other arrives unmatched and the
    page reports no price for a game it has prices for.
    """
    p_home = outcome_model.win_probability(features)
    sigma = uncertainty.estimate(outcome_model, features)

    # The same sigma applies to both sides: they are one number and its
    # complement, so an error in one is exactly an error in the other.
    out: dict[tuple[str, str], Belief] = {}
    for names, probability in (
        ((home, *home_aliases), p_home),
        ((away, *away_aliases), 1.0 - p_home),
    ):
        for name in names:
            if not name:
                continue
            out[(market, name.strip().lower(), None)] = Belief(
                probability=probability, sigma=sigma.value,
                measured=sigma.measured, basis=BASIS.get(market, ""))
    return out


def total(simulation: dict[str, Any], line: float,
          calibration: dict | None = None) -> dict[tuple[str, str], Belief]:
    """P(total over the posted number), from the simulated distribution.

    Half-integer lines only in practice, but whole numbers are handled: a total
    landing exactly on the number is a push, which is neither a win nor a loss,
    so those draws are removed from the denominator rather than counted as
    either. Counting a push as a loss would understate the over by the entire
    probability mass sitting on the line, which on a total of 9 is not small.
    """
    distribution = _weights(simulation.get("total_distribution"))
    if not distribution:
        return {}

    over = pushed = under = 0.0
    for runs, weight in distribution.items():
        if runs > line:
            over += weight
        elif runs < line:
            under += weight
        else:
            pushed += weight

    live = over + under
    if live <= 0:
        return {}
    p_over = over / live

    sigma = uncertainty.market_sigma(
        calibration or {}, types.TOTAL, p_over, line)
    return {
        (types.TOTAL, "over", line): Belief(
            probability=p_over, sigma=sigma or uncertainty.MINIMUM_SIGMA,
            measured=sigma is not None, basis=BASIS[types.TOTAL]),
        (types.TOTAL, "under", line): Belief(
            probability=1.0 - p_over, sigma=sigma or uncertainty.MINIMUM_SIGMA,
            measured=sigma is not None, basis=BASIS[types.TOTAL]),
    }


def runline(
    simulation: dict[str, Any], line: float, *, home: str, away: str,
    home_aliases: tuple[str, ...] = (), away_aliases: tuple[str, ...] = (),
    calibration: dict | None = None,
) -> dict[tuple[str, str], Belief]:
    """P(the home side covers the run line), from the simulated margin.

    The number is posted from the home side: a home favorite quoted -1.5 has to
    win by two, and the away side at +1.5 covers by losing by one or by winning
    outright. Read off the margin distribution, which comes from the same
    simulation as the win probability, so the two cannot disagree with each
    other.

    Calibrated against the totals record. Both are the same score model read a
    different way, and the run line has no separate measurement of its own --
    which is worth stating rather than implying a record that does not exist.
    """
    margins = _weights(simulation.get("margin_distribution"), key="margin")
    if not margins:
        return {}

    # The home side covers when its margin beats the number it is spotting,
    # which is `margin > -line`, not `margin > line`. A home favorite is posted
    # -1.5 and has to win by two: with the sign the wrong way round that read as
    # "wins by more than minus one and a half", which is every win and half the
    # losses. It made Cleveland 73.5% to cover -1.5 while the same model had
    # them winning the game 57.9% -- a team covering a spread more often than it
    # wins at all.
    covers = sum(w for m, w in margins.items() if m > -line)
    against = sum(w for m, w in margins.items() if m < -line)
    live = covers + against
    if live <= 0:
        return {}
    p_home = covers / live
    # Read against the total's record at the nearest whole number of runs.
    # The run line is the same score model seen from a different angle and
    # has no measurement of its own, which is worth saying rather than
    # implying a record that does not exist.
    sigma = uncertainty.market_sigma(
        calibration or {}, types.TOTAL, p_home, abs(line))

    # Each side is keyed with the number posted to *it*. A home favorite is
    # quoted -1.5 and the away side +1.5, so keying both on the home figure left
    # the away price unmatched -- a market with prices reporting as one without.
    out: dict[tuple[str, str], Belief] = {}
    for names, probability, posted in (
        ((home, *home_aliases), p_home, line),
        ((away, *away_aliases), 1.0 - p_home, -line),
    ):
        for name in names:
            if not name:
                continue
            out[(types.RUNLINE, name.strip().lower(), posted)] = Belief(
                probability=probability,
                sigma=sigma or uncertainty.MINIMUM_SIGMA,
                measured=sigma is not None, basis=BASIS[types.RUNLINE])
    return out


def _weights(distribution, key: str = "total") -> dict[float, float]:
    """Normalize either shape a distribution arrives in.

    The simulator emits a list of {"total": n, "p": weight} rows; a mapping of
    n to weight is the obvious thing to assume and is what the first version of
    this read. Handling only the assumed shape raised on a list and took the
    whole betting section out of the build -- caught only because the section is
    guarded, which is exactly how a silent version of this would have survived.
    """
    if not distribution:
        return {}
    if isinstance(distribution, dict):
        return {float(k): float(v) for k, v in distribution.items()}

    out: dict[float, float] = {}
    for row in distribution:
        if not isinstance(row, dict):
            continue
        value = row.get(key, row.get("total", row.get("runs", row.get("value"))))
        weight = row.get("p", row.get("probability", row.get("weight")))
        if value is None or weight is None:
            continue
        out[float(value)] = float(weight)
    return out


def first_five(
    projection, *, home: str, away: str,
    home_aliases: tuple[str, ...] = (), away_aliases: tuple[str, ...] = (),
    calibration: dict | None = None,
) -> dict[tuple[str, str], Belief]:
    """First-five moneyline, with the tie removed.

    F5 is quoted three ways at most books but two ways on the main line, where
    a tie after five is a push. Same treatment as a total landing on the
    number: the tied mass leaves the denominator instead of being handed to one
    side.
    """
    if projection is None:
        return {}
    home_leads = float(getattr(projection, "home_leads", 0.0))
    away_leads = float(getattr(projection, "away_leads", 0.0))
    live = home_leads + away_leads
    if live <= 0:
        return {}
    p_home = home_leads / live
    sigma = uncertainty.market_sigma(
        calibration or {}, types.F5_MONEYLINE, p_home)

    out: dict[tuple[str, str], Belief] = {}
    for names, probability in (
        ((home, *home_aliases), p_home),
        ((away, *away_aliases), 1.0 - p_home),
    ):
        for name in names:
            if not name:
                continue
            out[(types.F5_MONEYLINE, name.strip().lower(), None)] = Belief(
                probability=probability,
                sigma=sigma or uncertainty.MINIMUM_SIGMA,
                measured=sigma is not None,
                basis=BASIS[types.F5_MONEYLINE])
    return out


def strikeouts(prop, line: float,
               calibration: dict | None = None) -> dict[tuple[str, str], Belief]:
    """P(starter goes over the posted strikeout number).

    `at_least(k)` is inclusive, so clearing a line of 5.5 means at least 6 --
    the ceiling of the line, not the line rounded. Getting that wrong by one
    moves the probability by a whole distribution bar, which on a strikeout
    prop is worth several points.
    """
    if prop is None or not getattr(prop, "distribution", None):
        return {}
    import math

    needed = math.floor(line) + 1
    p_over = float(prop.at_least(needed))
    name = (getattr(prop, "name", "") or "").strip().lower()
    if not name:
        return {}

    sigma = uncertainty.market_sigma(
        calibration or {}, types.STRIKEOUTS, p_over, line)
    out: dict[tuple[str, str], Belief] = {}
    for alias in name_aliases(name):
        out[(types.STRIKEOUTS, f"{alias} over", line)] = Belief(
            probability=p_over, sigma=sigma or uncertainty.MINIMUM_SIGMA,
            measured=sigma is not None, basis=BASIS[types.STRIKEOUTS])
        out[(types.STRIKEOUTS, f"{alias} under", line)] = Belief(
            probability=1.0 - p_over, sigma=sigma or uncertainty.MINIMUM_SIGMA,
            measured=sigma is not None, basis=BASIS[types.STRIKEOUTS])
    return out


def strip_accents(text: str) -> str:
    """The same name without its diacritics.

    Rosters spell a player the way he spells himself and sportsbooks mostly do
    not. We carry "Walbert Urena" with a tilde and the board posts "walbert
    urena" without one, so an exact match found nothing and his whole strikeout
    market arrived unpriced -- a market with prices reporting as one without.
    """
    import unicodedata

    decomposed = unicodedata.normalize("NFKD", text)
    return "".join(c for c in decomposed if not unicodedata.combining(c))


def name_aliases(name: str) -> list[str]:
    """Every way a player's name might reasonably be written.

    The projection carries "tanner bibee" and a person types "Bibee", so
    matching on the full name alone silently prices nothing. The surname is the
    form actually used, and registering only the full name meant the whole
    strikeout market arrived as unmatched. Accent-stripped spellings are
    registered for the same reason.

    Ambiguity is resolved by the caller, not here: two starters sharing a
    surname must not both answer to it, and this function cannot see the other
    one.
    """
    name = " ".join(name.split()).lower()
    if not name:
        return []

    aliases: list[str] = []
    for spelling in (name, strip_accents(name)):
        if spelling and spelling not in aliases:
            aliases.append(spelling)
        parts = spelling.split(" ")
        if len(parts) > 1 and parts[-1] not in aliases:
            aliases.append(parts[-1])
    return aliases


def batter_prop(prop, market: str, line: float,
                calibration: dict | None = None) -> dict[tuple[str, str], Belief]:
    """P(a batter clears his posted number) for hits or home runs.

    The distribution is over a whole game rather than a fixed number of plate
    appearances -- how many turns a hitter gets is itself uncertain and already
    folded in -- so this only has to read the tail.

    Clearing 0.5 means at least one, clearing 1.5 means at least two: the
    ceiling of the line, not the line rounded. Every book posts hits and home
    runs at half-integers, so there is never a push to remove.
    """
    if prop is None:
        return {}
    import math

    field = {types.HITS: "hits", types.HOME_RUNS: "home_runs"}.get(market)
    if field is None:
        return {}
    block = getattr(prop, field, None) or {}
    distribution = block.get("distribution") or {}
    if not distribution:
        return {}

    needed = math.floor(line) + 1
    p_over = float(sum(
        weight for count, weight in distribution.items() if int(count) >= needed))
    name = (getattr(prop, "name", "") or "").strip().lower()
    if not name:
        return {}

    sigma = uncertainty.market_sigma(calibration or {}, market, p_over, line)
    out: dict[tuple[str, str], Belief] = {}
    for alias in name_aliases(name):
        out[(market, f"{alias} over", line)] = Belief(
            probability=p_over, sigma=sigma or uncertainty.MINIMUM_SIGMA,
            measured=sigma is not None, basis=BASIS[market])
        out[(market, f"{alias} under", line)] = Belief(
            probability=1.0 - p_over, sigma=sigma or uncertainty.MINIMUM_SIGMA,
            measured=sigma is not None, basis=BASIS[market])
    return out
