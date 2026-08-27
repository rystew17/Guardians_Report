"""Everything the page shows, assembled once so the template only formats.

The section is deliberately flat: one row per priced selection, carrying every
number that went into its verdict. A reader who disagrees with a conclusion
should be able to recompute it by hand from the row itself rather than having to
trust the total, which is the same rule the rest of this report follows.

Both sides of every market are shown, not only the side we lean. Printing only
the attractive half would hide that the other half is the same disagreement
with its sign flipped, and would make a 6-point gap look like a discovery rather
than an arithmetic necessity.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from guards_report.betting import clv as clv_module, verdict as verdict_module
from guards_report.betting.edge import Play
from guards_report.betting.guide import Belief, Night
from guards_report.odds import types

MARKET_LABELS = {
    types.MONEYLINE: "Moneyline",
    types.TOTAL: "Total runs",
    types.RUNLINE: "Run line",
    types.F5_MONEYLINE: "First five",
    types.F5_TOTAL: "First five total",
    types.STRIKEOUTS: "Strikeouts",
    types.HITS: "Hits",
    types.HOME_RUNS: "Home runs",
    types.TOTAL_BASES: "Total bases",
}


def _subject(selection: str) -> str:
    """The name a bet is about, ignoring side and number."""
    text = selection.strip().lower()
    for side in (" over", " under"):
        if text.endswith(side):
            return text[: -len(side)]
    return text


# Below this an edge is not worth marking. Half a point is inside the rounding
# the page prints at, so anything under it would highlight a cell whose numbers
# read as identical.
NOTEWORTHY_EDGE = 0.005


def american(value: float) -> str:
    """As a book prints it, with the plus that a bare number loses."""
    return f"{value:+.0f}"


@dataclass(frozen=True)
class Row:
    """One priced selection, with its verdict and everything behind it."""

    market: str
    market_label: str
    selection: str
    line: float | None
    american: str
    p_model: float
    p_market: float
    break_even: float
    disagreement: float
    edge: float
    sigma: float
    z: float
    confidence: float
    stake: float
    basis: str
    verdict: verdict_module.Verdict
    subject: str = ""      # the player, for markets quoted per player
    team: str = ""
    slot: int | None = None

    @property
    def is_bet(self) -> bool:
        return self.verdict.is_bet

    @property
    def side(self) -> str:
        """over | under, for a market that has sides."""
        text = self.selection.strip().lower()
        for word in ("over", "under"):
            if text.endswith(word):
                return word
        return ""

    @property
    def tone(self) -> str:
        """How strongly the page should mark this cell.

        Only two states earn a color: one we would bet, and one that cleared the
        price but not our own error. Everything else is the ordinary case and
        colouring it would drown the two that matter.
        """
        if self.verdict.is_bet:
            return "bet"
        # An edge that rounds to +0.0 is not "close", whatever the verdict says
        # about its sign. Marking it implies a near miss where there is only a
        # rounding artifact, and on a full board that is most of the colour.
        if (self.verdict.action == verdict_module.PASS_INSIDE_ERROR
                and self.edge >= NOTEWORTHY_EDGE):
            return "near"
        return ""


@dataclass
class Section:
    """The whole block, including what it could not do."""

    disclaimer: str = verdict_module.DISCLAIMER
    rows: list[Row] = field(default_factory=list)
    summary: str = ""
    tau: float = 0.0
    z_threshold: float = 0.0
    devig: str = "shin"
    record: Any | None = None
    warnings: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    unmatched: list[str] = field(default_factory=list)
    assumed_margin: list[str] = field(default_factory=list)
    teams: tuple = ()
    sigma_note: str = ""

    @property
    def bets(self) -> list[Row]:
        return [r for r in self.rows if r.is_bet]

    @property
    def has_prices(self) -> bool:
        return bool(self.rows)

    @property
    def featured(self) -> Row | None:
        """The row worth reading in full -- only ever an actual bet.

        A near miss used to be promoted here when nothing cleared, and it read
        as a recommendation. On a night when we made Cleveland 58.1% at a price
        needing 61.7%, the only positive-value side was the Angels at +150 --
        so the page led with the team we expect to lose, under a heading that
        looked like a pick. The arithmetic was right and the presentation was
        indefensible.

        Nothing is promoted unless it is a bet. A quiet night says so at the
        top and leaves the rest in its group.
        """
        found = [r for r in self.rows if r.verdict.action == verdict_module.BET]
        return found[0] if found else None

    # Markets a reader thinks about together. A game bet and a hitter prop are
    # different questions and belong in different tables; the first version put
    # forty of them in one list sorted by edge, which is a dump, not a page.
    FAMILIES = (
        ("The game", (types.MONEYLINE, types.RUNLINE, types.TOTAL,
                      types.F5_MONEYLINE, types.F5_TOTAL)),
        ("Starting pitchers", (types.STRIKEOUTS,)),
        ("Hitters", (types.HITS, types.HOME_RUNS, types.TOTAL_BASES)),
    )

    @property
    def lineups(self) -> list[dict]:
        """One table per side, laid out like the projections lineup card.

        A hitter is one row and his bets are columns, which is how the rest of
        this report already presents a lineup. Flattened into a list sorted by
        edge, the same nine names appeared four times each in no order anyone
        reads a lineup in, and fifty rows said less than nine do.
        """
        players: dict[str, dict] = {}
        for row in self.rows:
            if row.market not in (types.HITS, types.HOME_RUNS) or not row.subject:
                continue
            entry = players.setdefault(row.subject, {
                "subject": row.subject,
                "name": row.selection.rsplit(" ", 1)[0],
                "team": row.team, "slot": row.slot,
                "hit_line": None, "hr_line": None,
                "hits_over": None, "hits_under": None,
                "hr_over": None, "hr_under": None,
            })
            field_name = {
                (types.HITS, "over"): "hits_over",
                (types.HITS, "under"): "hits_under",
                (types.HOME_RUNS, "over"): "hr_over",
                (types.HOME_RUNS, "under"): "hr_under",
            }.get((row.market, row.side))
            if field_name:
                entry[field_name] = row
                # Carried so the column can name its own number. "HR over" with
                # no line reads as one bet whichever line produced it, which is
                # how a two-homer price passed for a to-homer price.
                key = "hit_line" if row.market == types.HITS else "hr_line"
                if row.line is not None:
                    entry[key] = row.line

        order = [t for t in self.teams if t] or sorted(
            {p["team"] for p in players.values() if p["team"]})
        out = []
        for team in order:
            found = [p for p in players.values() if p["team"] == team]
            if not found:
                continue
            # Batting order where it is known, alphabetical where it is not --
            # which is the honest presentation when the card has not posted.
            found.sort(key=lambda p: (p["slot"] is None, p["slot"] or 0, p["name"]))
            out.append({
                "team": team,
                "players": found,
                "carded": any(p["slot"] for p in found),
                "hit_line": next(
                    (p["hit_line"] for p in found if p["hit_line"] is not None), None),
                "hr_line": next(
                    (p["hr_line"] for p in found if p["hr_line"] is not None), None),
                # Books post home runs to happen and not to not happen, so the
                # under column is usually empty. An always-blank column is
                # noise, so it appears only when something is in it.
                "hr_under": any(p["hr_under"] for p in found),
                "bets": sum(
                    1 for p in found
                    for cell in (p["hits_over"], p["hits_under"],
                                 p["hr_over"], p["hr_under"])
                    if cell is not None and cell.is_bet),
            })

        loose = [p for p in players.values() if not p["team"]]
        if loose:
            loose.sort(key=lambda p: p["name"])
            out.append({"team": "", "players": loose, "carded": False,
                        "hit_line": next(
                            (p["hit_line"] for p in loose
                             if p["hit_line"] is not None), None),
                        "hr_line": next(
                            (p["hr_line"] for p in loose
                             if p["hr_line"] is not None), None),
                        "hr_under": any(p["hr_under"] for p in loose),
                        "bets": sum(
                            1 for p in loose
                            for cell in (p["hits_over"], p["hits_under"],
                                         p["hr_over"], p["hr_under"])
                            if cell is not None and cell.is_bet)})
        return out

    @property
    def pitchers(self) -> list[dict]:
        """One table per starter rather than both stacked together.

        Two pitchers in one table read as a single list of eight strikeout
        prices with no indication which four belong to whom.
        """
        found: dict[str, list[Row]] = {}
        for row in self.rows:
            if row.market != types.STRIKEOUTS or not row.subject:
                continue
            found.setdefault(row.subject, []).append(row)

        out = []
        for subject, rows in found.items():
            rows.sort(key=lambda r: (r.line or 0, r.side))
            out.append({
                "subject": subject,
                "name": rows[0].selection.rsplit(" ", 1)[0],
                "team": rows[0].team,
                "rows": rows,
                "bets": sum(1 for r in rows if r.is_bet),
            })
        out.sort(key=lambda p: (-p["bets"], p["name"]))
        return out

    @property
    def game_rows(self) -> list[Row]:
        """Everything that is about the game rather than about a player."""
        markets = (types.MONEYLINE, types.RUNLINE, types.TOTAL,
                   types.F5_MONEYLINE, types.F5_TOTAL)
        order = {verdict_module.BET: 0, verdict_module.PASS_DUPLICATE: 1,
                 verdict_module.PASS_INSIDE_ERROR: 2,
                 verdict_module.PASS_UNMEASURED: 3,
                 verdict_module.PASS_PRICED_IN: 4}
        found = [r for r in self.rows if r.market in markets]
        found.sort(key=lambda r: (order.get(r.verdict.action, 9), -r.edge))
        return found

    @property
    def families(self) -> list[dict]:
        """Rows split by what a reader is actually deciding between.

        Within a family the verdict still leads, because "can I bet this" is the
        question and the answer sorts the rows. Across families nothing is
        compared, because a moneyline and a hitter's total bases are not
        alternatives to each other.
        """
        order = {verdict_module.BET: 0, verdict_module.PASS_DUPLICATE: 1,
                 verdict_module.PASS_INSIDE_ERROR: 2,
                 verdict_module.PASS_UNMEASURED: 3,
                 verdict_module.PASS_PRICED_IN: 4}
        out = []
        for heading, markets in self.FAMILIES:
            members = [r for r in self.rows if r.market in markets]
            if not members:
                continue
            members.sort(key=lambda r: (order.get(r.verdict.action, 9), -r.edge))
            reasons: list[str] = []
            for row in members:
                if row.verdict.reason not in reasons:
                    reasons.append(row.verdict.reason)
            out.append({
                "heading": heading,
                "rows": members,
                "playable": [r for r in members if r.is_bet],
                "reasons": reasons,
            })
        return out

    @property
    def groups(self) -> list[dict]:
        """The remaining rows, gathered under one shared explanation each.

        The reason text depends only on the verdict, so it belongs to the group
        rather than to every row inside it. The numbers that do vary are already
        in the columns.
        """
        featured = self.featured
        order = [
            (verdict_module.BET, "Worth a bet"),
            (verdict_module.PASS_DUPLICATE, "Already covered by another bet"),
            (verdict_module.PASS_INSIDE_ERROR, "Close, but inside our own error"),
            (verdict_module.PASS_UNMEASURED, "No record yet"),
            (verdict_module.PASS_PRICED_IN, "Already priced in"),
        ]
        out = []
        for action, heading in order:
            members = [
                r for r in self.rows
                if r.verdict.action == action and r is not featured
            ]
            if not members:
                continue
            out.append({
                "action": action,
                "heading": heading,
                "reason": members[0].verdict.reason,
                "rows": members,
            })
        return out


# Why sigma is what it is. Worth stating on the page rather than burying,
# because it is a limit of the evidence rather than of the model: with 11,668
# held-out games, miscalibration below about 1.4 points cannot be told from
# sampling noise, and the standard error is floored there. More seasons would
# lower the floor and let more disagreements through.
SIGMA_NOTE = (
    "Our standard error is currently set by how much history we have, not by "
    "how good the model is. Across {n:,} held-out games, miscalibration below "
    "{floor:.1f} points cannot be separated from sampling noise, so that is "
    "the floor. More seasons would tighten it and let smaller edges through."
)


def build(
    night: Night,
    beliefs: dict[tuple[str, str], Belief],
    *,
    tau: float,
    z_threshold: float,
    devig: str = "shin",
    record: Any | None = None,
    lines: dict[tuple[str, str], float] | None = None,
    calibration: list[dict] | None = None,
    roster: dict[str, tuple[str, int | None]] | None = None,
    teams: tuple = (),
) -> Section:
    """Turn a priced night into the rows the template renders."""
    lines = lines or {}
    roster = roster or {}
    rows: list[Row] = []

    staked = {(p.market, p.selection, p.line) for p in night.plays}
    strongest: dict[tuple[str, str], Any] = {}
    for play in night.plays:
        strongest[(play.market, _subject(play.selection))] = play

    for play in night.considered:
        key = (play.market, play.selection.lower(), play.line)
        belief = beliefs.get(key)
        measured = bool(belief and belief.measured)
        basis = belief.basis if belief else ""

        # A play the selector dropped as a duplicate position must not still be
        # badged as a bet: only one of them carries a stake.
        covering = strongest.get((play.market, _subject(play.selection)))
        superseded = ""
        if (covering is not None
                and (play.market, play.selection, play.line) not in staked
                and play.expected_value > 0 and play.z >= z_threshold
                and measured):
            superseded = f"{covering.selection} at {covering.american:+.0f}"

        rows.append(Row(
            market=play.market,
            market_label=MARKET_LABELS.get(play.market, play.market.title()),
            selection=play.selection,
            line=play.line if play.line is not None else lines.get(key),
            american=american(play.american),
            p_model=play.p_model,
            p_market=play.p_market,
            break_even=play.break_even,
            disagreement=play.disagreement,
            edge=play.edge,
            sigma=play.sigma,
            z=play.z,
            confidence=play.confidence,
            stake=(play.stake if measured and not superseded else 0.0),
            basis=basis,
            subject=_subject(play.selection),
            team=roster.get(_subject(play.selection), ("", None))[0],
            slot=roster.get(_subject(play.selection), ("", None))[1],
            verdict=verdict_module.decide(
                play, measured=measured, z_threshold=z_threshold, basis=basis,
                superseded_by=superseded),
        ))

    # Bets first, then near misses, then everything else -- so the reader meets
    # the decision before the arithmetic. Within a group, by confidence.
    order = {verdict_module.BET: 0, verdict_module.PASS_DUPLICATE: 1,
             verdict_module.PASS_INSIDE_ERROR: 2,
             verdict_module.PASS_UNMEASURED: 3, verdict_module.PASS_PRICED_IN: 4}
    rows.sort(key=lambda r: (order.get(r.verdict.action, 9), -r.z))

    return Section(
        rows=rows,
        summary=verdict_module.summarize([r.verdict for r in rows]),
        tau=tau,
        z_threshold=z_threshold,
        devig=devig,
        record=record,
        warnings=list(night.warnings),
        unmatched=list(night.unmatched),
        assumed_margin=list(night.assumed_margin),
        teams=teams or (),
        sigma_note=_sigma_note(calibration),
    )


def _sigma_note(calibration: list[dict] | None) -> str:
    if not calibration:
        return ""
    games = sum(int(entry.get("n", 0)) for entry in calibration)
    floors = [float(entry.get("resolution", 0.0)) for entry in calibration]
    if not games or not floors:
        return ""
    return SIGMA_NOTE.format(n=games, floor=(sum(floors) / len(floors)) * 100)
