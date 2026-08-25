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
from guards_report.odds import client, store, types

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

    notes: list[str] = []

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
    elif client.configured():
        # Pulled rather than typed, when a key is present. Failures are
        # reported and never fall back to an older capture -- comparing
        # tonight's projection against last week's line is worse than showing
        # nothing.
        try:
            pull = client.fetch(
                game_date=game_date,
                home_team=bundle.home.name,
                away_team=bundle.away.name,
            )
            if pull.markets:
                store.append(root, [q for m in pull.markets for q in m.quotes])
                notes.append(
                    f"Prices pulled from {pull.markets[0].book}"
                    + (f", {pull.credits_remaining} API credits left this month"
                       if pull.credits_remaining is not None else ""))
            if pull.note:
                notes.append(pull.note)
        except client.OddsAPIError as exc:
            notes.append(f"Could not pull odds: {exc}")

    markets = store.latest_markets(root, game_date)
    if not markets:
        if notes:
            bundle.betting = section.Section(warnings=notes)
            return bundle.betting
        return None

    measured = client_calibration(root)
    beliefs = _beliefs(bundle, markets, root, measured)
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

    built = section.build(
        night, beliefs, tau=TAU, z_threshold=Z_THRESHOLD, devig=DEVIG,
        record=record, lines=lines, calibration=calibration,
        roster=_roster(bundle, markets, root),
        teams=(getattr(bundle.away, "abbreviation", "") or bundle.away.name,
               getattr(bundle.home, "abbreviation", "") or bundle.home.name))
    # Where the prices came from is information, not a problem. Filing it under
    # warnings made "pulled from DraftKings" read as something to check.
    built.notes = notes
    bundle.betting = built
    return bundle.betting


def _beliefs(bundle, markets, root: Path, calibration: dict | None = None) -> dict:
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
    # Typed prices use the abbreviation, the feed uses the full club name.
    home_aliases = (bundle.home.name,)
    away_aliases = (bundle.away.name,)

    calibration = calibration or {}
    beliefs: dict = {}

    features = getattr(projection, "win_features", None)
    outcome = _fitted_model(root)
    if outcome is not None and features:
        beliefs.update(sources.moneyline(
            outcome, features, home=home, away=away,
            home_aliases=home_aliases, away_aliases=away_aliases))

    score = getattr(projection, "score", None)
    if isinstance(score, dict):
        for line in _lines_for(markets, types.TOTAL):
            beliefs.update(sources.total(score, line, calibration))
        for market in markets:
            if market.name != types.RUNLINE:
                continue
            # Signed from the home side: a home favorite is posted -1.5 and has
            # to win by two. Taking the magnitude would price the wrong side.
            posted = None
            for candidate in (home, *home_aliases):
                posted = market.line_for(candidate)
                if posted is not None:
                    break
            if posted is None:
                continue
            beliefs.update(sources.runline(
                score, posted, home=home, away=away,
                home_aliases=home_aliases, away_aliases=away_aliases,
                calibration=calibration))

    f5 = getattr(projection, "first_five", None)
    beliefs.update(sources.first_five(
        f5, home=home, away=away,
        home_aliases=home_aliases, away_aliases=away_aliases,
        calibration=calibration))
    for line in _lines_for(markets, types.F5_TOTAL):
        beliefs.update(sources.first_five_total(f5, line, calibration))

    # Each prop is priced against *its own* market's line, resolved market by
    # market rather than by pairing every pitcher with every line on the board.
    # The cartesian version wrote the same key once per line and the last one
    # won, so on a night with two starters posted at 4.5 and 7.5 the pitcher
    # listed at 4.5 was priced against 7.5 -- his chance of going over came out
    # at 2.9% instead of 35%, which reads as a 42-point disagreement with the
    # book rather than as a bug.
    props = _starter_props(projection)
    ambiguous = _shared_surnames(props)
    for market in markets:
        if market.name != types.STRIKEOUTS or market.line is None:
            continue
        for prop in props:
            if not _names_this_market(prop, market, ambiguous):
                continue
            for key, belief in sources.strikeouts(
                    prop, market.line, calibration).items():
                if any(key[1].startswith(f"{surname} ") for surname in ambiguous):
                    continue
                if key[1] in {q.selection.strip().lower() for q in market.quotes}:
                    beliefs[key] = belief

    # Batter props, resolved the same way: each hitter against his own posted
    # number, never the cartesian product of every hitter and every line.
    batters = _batter_props(projection)
    # The card is often not out yet; the board always is.
    batters = batters + _fill_missing_hitters(bundle, markets, root)
    batter_ambiguous = _shared_surnames(batters)
    for market in markets:
        if market.name not in (types.HITS, types.HOME_RUNS) or market.line is None:
            continue
        posted = {q.selection.strip().lower() for q in market.quotes}
        for prop in batters:
            if not _names_this_market(prop, market, batter_ambiguous):
                continue
            for key, belief in sources.batter_prop(
                    prop, market.name, market.line, calibration).items():
                if any(key[1].startswith(f"{s} ") for s in batter_ambiguous):
                    continue
                if key[1] in posted:
                    beliefs[key] = belief

    return beliefs


def _batter_props(projection) -> list:
    """Every hitter projected tonight, both lineups."""
    block = getattr(projection, "player_props", None)
    if not isinstance(block, dict):
        return []
    found = []
    for value in block.values():
        if value is None:
            continue
        found.extend(value if isinstance(value, (list, tuple)) else [value])
    return found


def _roster(bundle, markets, root: Path) -> dict:
    """Which side each priced player bats for, and where in the order.

    Taken from the report's own team sections rather than from the odds feed,
    which does not say. Without it the hitters cannot be laid out as two
    lineups, only as one undifferentiated list.
    """
    from guards_report.betting import sources as _sources

    out: dict[str, tuple[str, int | None]] = {}
    slots: dict[str, int] = {}
    for prop in _batter_props(getattr(bundle, "projection", None)):
        name = _sources.strip_accents(
            (getattr(prop, "name", "") or "").strip().lower())
        if name and getattr(prop, "slot", None):
            slots[name] = int(prop.slot)

    for section_ in (bundle.away, bundle.home):
        team = getattr(section_, "abbreviation", "") or section_.name
        for player in list(getattr(section_, "batters", []) or []) + list(
                getattr(section_, "pitchers", []) or []):
            plain = _sources.strip_accents((player.name or "").strip().lower())
            if not plain:
                continue
            out.setdefault(plain, (team, slots.get(plain)))
            surname = plain.split(" ")[-1]
            out.setdefault(surname, (team, slots.get(plain)))
    return out


def _posted_hitters(markets) -> set[str]:
    """Every hitter the board has priced, accent-free."""
    names: set[str] = set()
    for market in markets:
        if market.name not in (types.HITS, types.HOME_RUNS):
            continue
        for quote in market.quotes:
            subject = quote.subject
            if subject:
                names.add(sources.strip_accents(subject))
    return names


def _fill_missing_hitters(bundle, markets, root: Path) -> list:
    """Project the hitters the board priced but the lineup card has not named.

    A lineup posts a few hours before first pitch and the prop board goes up
    well before that, so for most of the day the page had prices for twenty
    hitters and a projection for none of them -- every one reported as "no
    projection matching this selection", which reads as a defect rather than as
    a card that is not out yet.

    The board is itself a lineup signal: a book does not price a hitter it does
    not expect to play. So the names it posts become the lineup, and each is
    projected from his own rates the same way a carded hitter is. Batting order
    is unknown, which only affects how many turns he is expected to get, and
    that uncertainty is already carried by the unknown-slot distribution.
    """
    posted = _posted_hitters(markets)
    if not posted:
        return []

    already = {
        sources.strip_accents((getattr(p, "name", "") or "").strip().lower())
        for p in _batter_props(getattr(bundle, "projection", None))
    }
    missing = posted - already
    if not missing:
        return []

    from guards_report.projections import tonight, train_props
    from guards_report.ingest.preview import _plate_appearances
    from guards_report.config import load_settings

    artifact = train_props.load_props(Path(root) / "models" / "props.json")
    if artifact is None:
        return []
    try:
        plate = _plate_appearances(load_settings(), bundle.game_date)
    except Exception:  # noqa: BLE001 -- never cost the report
        return []
    if plate is None or not len(plate):
        return []

    out = []
    for section, other in ((bundle.home, bundle.away), (bundle.away, bundle.home)):
        wanted, names, stands = [], {}, {}
        for batter in getattr(section, "batters", []) or []:
            plain = sources.strip_accents((batter.name or "").strip().lower())
            if plain not in missing:
                continue
            wanted.append(int(batter.player_id))
            names[int(batter.player_id)] = batter.name
            stands[int(batter.player_id)] = getattr(batter, "bat_side", "R") or "R"
        if not wanted:
            continue

        starter = next(
            (p for p in getattr(other, "pitchers", []) or []
             if getattr(p, "is_probable_starter", False)), None)
        try:
            out.extend(tonight.batter_props(
                artifact, plate, on=bundle.game_date, lineup=wanted, names=names,
                opposing_starter=getattr(starter, "player_id", None),
                opposing_throws=getattr(starter, "hand", "R") or "R",
                stands=stands,
                home_team=getattr(bundle.home, "abbreviation", "") or "",
            ) or [])
        except Exception:  # noqa: BLE001 -- a missing projection is not fatal
            continue
    return out


def _names_this_market(prop, market, ambiguous: set[str]) -> bool:
    """Whether this market is quoting this pitcher.

    Matched on the selection text the book sent, which carries the player's
    name. Without the check every prop would be priced against every line on
    the board.
    """
    name = (getattr(prop, "name", "") or "").strip().lower()
    if not name:
        return False
    aliases = set(sources.name_aliases(name)) - ambiguous
    for quote in market.quotes:
        selection = quote.selection.strip().lower()
        plain = sources.strip_accents(selection)
        if any(selection.startswith(f"{alias} ") or plain.startswith(f"{alias} ")
               for alias in aliases):
            return True
    return False


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


def client_calibration(root: Path) -> dict:
    """Every market's record against outcomes, measured by scripts/calibrate.py.

    Absent, every market outside the moneyline reads as unmeasured and cannot be
    staked -- which is the correct behavior, and was the only behavior before
    this file existed.
    """
    from guards_report.betting import uncertainty

    return uncertainty.load_market_calibration(root)


def _fitted_model(root: Path):
    from guards_report.projections import model as model_module

    return model_module.load(Path(root) / "models" / "game_outcome.json")


