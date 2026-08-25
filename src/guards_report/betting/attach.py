"""Wiring the betting block onto a finished report bundle.

Runs last, after every number it compares against is final, and never blocks a
build. A report without prices is a complete report -- the same rule the
projection and analysis layers follow -- so every failure here degrades to no
section rather than to no report.

Prices are stored before they are shown. A play that reached the page without
reaching the record is a play whose closing line can never be checked, which
would quietly remove the only scoreboard this project has.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any

from guards_report.betting import clv, guide, section, sources, verdict
from guards_report.odds import store, types

# Chosen deliberately and shown on the page rather than hidden. Tau says how
# wrong the closing line typically is, which cannot be measured without a record
# of closing lines -- so it is an assumption until the CLV record can replace it.
TAU = 0.03
Z_THRESHOLD = 2.5
DEVIG = "shin"


def attach(
    bundle,
    *,
    root: Path,
    odds_text: str = "",
    book: str = "manual",
) -> Any | None:
    """Price tonight's markets against our projections. Returns the section."""
    game_date: date = bundle.game_date

    if odds_text.strip():
        try:
            store.record(root, odds_text, game_date=game_date,
                         book=book, game_pk=bundle.game_pk)
        except ValueError as exc:
            # A typo in hand-typed prices must name itself rather than
            # silently producing a page with one side of a market missing.
            bundle.betting = section.Section(
                warnings=[f"Could not read the prices: {exc}"])
            return bundle.betting

    markets = store.latest_markets(root, game_date)
    if not markets:
        return None

    beliefs = _beliefs(bundle, markets, root)
    if not beliefs:
        bundle.betting = section.Section(
            warnings=["No fitted projection for this game, so nothing to "
                      "compare the prices against."])
        return bundle.betting

    night = guide.build(
        markets, beliefs, tau=TAU, z_threshold=Z_THRESHOLD, devig=DEVIG)

    # Everything considered is logged, not only what cleared. A record holding
    # bets alone cannot answer whether the bar is set correctly, because it
    # contains no examples of what was turned down.
    actions = {
        f"{p.market}:{p.selection}": "bet" for p in night.plays
    }
    try:
        clv.log(root, night.considered, game_date=game_date,
                actions=actions, book=book)
    except Exception as exc:  # noqa: BLE001 -- logging must not cost the report
        night.warnings.append(f"Could not log tonight's plays: {exc}")

    record = clv.summarize(
        clv.measure(root, clv.closing_probabilities(markets, devig=DEVIG),
                    game_date=game_date))

    lines = {
        (m.name, q.selection.lower()): q.line
        for m in markets for q in m.quotes if q.line is not None
    }

    fitted = _fitted_model(root)
    calibration = getattr(fitted, "win_calibration", None) if fitted else None

    bundle.betting = section.build(
        night, beliefs, tau=TAU, z_threshold=Z_THRESHOLD, devig=DEVIG,
        record=record, lines=lines, calibration=calibration)
    return bundle.betting


def _beliefs(bundle, markets, root: Path) -> dict:
    """What we think about each outcome, gathered from whichever models ran.

    Only the moneyline carries a measured standard error, because it is the only
    model with a calibration record behind it. The rest are priced and shown but
    cannot be staked, which `guide.build` enforces -- see `betting.sources` for
    why that is a real constraint rather than a stub.

    Lines come from the markets actually entered, not from a guessed set. Asking
    for the probability of a total of 8.5 when the book posted 9 would produce a
    number that looks fine and describes a different bet.
    """
    projection = getattr(bundle, "projection", None)
    if projection is None:
        return {}

    home = getattr(bundle.home, "abbreviation", "") or bundle.home.name
    away = getattr(bundle.away, "abbreviation", "") or bundle.away.name

    beliefs: dict = {}

    features = getattr(projection, "win_features", None)
    outcome = _fitted_model(root)
    if outcome is not None and features:
        beliefs.update(sources.moneyline(outcome, features, home=home, away=away))

    score = getattr(projection, "score", None)
    if isinstance(score, dict):
        for line in _lines_for(markets, types.TOTAL):
            beliefs.update(sources.total(score, line))

    beliefs.update(sources.first_five(
        getattr(projection, "first_five", None), home=home, away=away))

    props = _starter_props(projection)
    ambiguous = _shared_surnames(props)
    for prop in props:
        for line in _lines_for(markets, types.STRIKEOUTS):
            for key, belief in sources.strikeouts(prop, line).items():
                # Both starters answering to the same surname would price one
                # pitcher's market with the other's distribution, and the page
                # would show a number rather than a problem.
                if any(key[1].startswith(f"{surname} ") for surname in ambiguous):
                    continue
                beliefs[key] = belief

    return beliefs


def _shared_surnames(props) -> set[str]:
    """Surnames belonging to more than one starter tonight."""
    seen: dict[str, int] = {}
    for prop in props:
        name = (getattr(prop, "name", "") or "").strip().lower()
        parts = name.split()
        if len(parts) > 1:
            seen[parts[-1]] = seen.get(parts[-1], 0) + 1
    return {surname for surname, count in seen.items() if count > 1}


def _lines_for(markets, market_name: str) -> list[float]:
    """The numbers actually posted for one market."""
    return sorted({
        q.line for m in markets if m.name == market_name
        for q in m.quotes if q.line is not None
    })


def _starter_props(projection) -> list:
    """Both starters' strikeout distributions, whichever side they are on."""
    block = getattr(projection, "strikeouts", None)
    if not isinstance(block, dict):
        return []
    found = []
    for value in block.values():
        if value is None:
            continue
        found.extend(value if isinstance(value, (list, tuple)) else [value])
    return found


def _fitted_model(root: Path):
    from guards_report.projections import model as model_module

    return model_module.load(Path(root) / "models" / "game_outcome.json")


