"""The "what to watch" block: rank the extremes already present in the data.

This is a sorting problem, not an analysis one. Every fact produced here is a
figure computed elsewhere in the report, selected because it is the largest or
smallest of its kind in today's matchup. Nothing is written, inferred, or
estimated -- which is what lets a block aimed at the casual reader sit inside a
report whose whole claim is that no number was generated.

Each highlight carries the number it was selected on, so a reader can always
see why it was surfaced rather than taking the selection on trust.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

# Minimum playing time before a player is eligible to be called an extreme.
# Without these the block would be a list of players with four at-bats.
MIN_PA_SEASON = 120
MIN_PA_RECENT = 25
MIN_PA_SPLIT = 25
MIN_BF = 60


@dataclass
class Highlight:
    """One surfaced fact, with the value it was chosen on."""

    kind: str
    headline: str
    detail: str
    value: str
    team: str


def _num(value: Any) -> float | None:
    if value in (None, "", "-", "–"):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _rate(value: float) -> str:
    """Format a rate the way the rest of the report does: .931, not 0.931."""
    text = f"{value:.3f}"
    return text[1:] if text.startswith("0.") else text


def _signed_rate(value: float) -> str:
    text = f"{value:+.3f}"
    return text.replace("+0.", "+.").replace("-0.", "-.")


def _window(box: Any, label: str) -> dict[str, Any]:
    for win in box.windows:
        if win.label == label:
            return win.stats
    return {}


def build(bundle: Any) -> list[Highlight]:
    """Assemble the highlight block for one game."""
    out: list[Highlight] = []
    teams = [bundle.guardians, bundle.opponent]

    # -- hottest and coldest bats over the last fifteen games ---------------
    recent: list[tuple[float, Any, str, int]] = []
    for team in teams:
        for box in team.batters:
            stats = _window(box, "L15")
            pa = stats.get("plateAppearances") or 0
            ops = _num(stats.get("ops"))
            if ops is not None and pa >= MIN_PA_RECENT:
                recent.append((ops, box, team.abbreviation, pa))

    if recent:
        recent.sort(key=lambda item: item[0], reverse=True)
        ops, box, abbr, pa = recent[0]
        out.append(
            Highlight(
                kind="hot",
                headline=f"{box.name} is the hottest bat in the game",
                detail=f"{_rate(ops)} OPS over his last 15 games ({pa} PA)",
                value=_rate(ops),
                team=abbr,
            )
        )
        ops, box, abbr, pa = recent[-1]
        out.append(
            Highlight(
                kind="cold",
                headline=f"{box.name} is the coldest",
                detail=f"{_rate(ops)} OPS over his last 15 games ({pa} PA)",
                value=_rate(ops),
                team=abbr,
            )
        )

    # -- largest platoon edge against today's starters ----------------------
    edges: list[tuple[float, Any, str, str, int]] = []
    for team in teams:
        for box in team.batters:
            if not box.vs_hand:
                continue
            pa = int(_num(box.vs_hand.get("plateAppearances")) or 0)
            split_ops = _num(box.vs_hand.get("ops"))
            season_ops = _num(box.season.get("ops"))
            if split_ops is None or season_ops is None or pa < MIN_PA_SPLIT:
                continue
            edges.append(
                (split_ops - season_ops, box, team.abbreviation,
                 box.vs_hand_label, pa)
            )

    if edges:
        edges.sort(key=lambda item: item[0], reverse=True)
        diff, box, abbr, label, pa = edges[0]
        out.append(
            Highlight(
                kind="edge",
                headline=f"{box.name} has the biggest platoon edge today",
                detail=(
                    f"{_rate(_num(box.vs_hand.get('ops')) or 0.0)} OPS {label} "
                    f"({pa} PA), {_signed_rate(diff)} against his overall line"
                ),
                value=_signed_rate(diff),
                team=abbr,
            )
        )
        diff, box, abbr, label, pa = edges[-1]
        if diff < 0:
            out.append(
                Highlight(
                    kind="mismatch",
                    headline=f"{box.name} is the worst matchup on paper",
                    detail=(
                        f"{_rate(_num(box.vs_hand.get('ops')) or 0.0)} OPS {label} "
                        f"({pa} PA), {_signed_rate(diff)} against his overall line"
                    ),
                    value=_signed_rate(diff),
                    team=abbr,
                )
            )

    # -- the starters, on recent form --------------------------------------
    for team in teams:
        starter = next((p for p in team.pitchers if p.is_probable_starter), None)
        if not starter:
            continue
        season_era = _num(starter.season.get("era"))
        recent_stats = _window(starter, "L5")
        recent_era = _num(recent_stats.get("era"))
        if season_era is None or recent_era is None:
            continue
        if recent_stats.get("games", 0) < 3:
            continue
        swing = recent_era - season_era
        if abs(swing) < 1.0:
            continue
        out.append(
            Highlight(
                kind="hot" if swing < 0 else "cold",
                headline=(
                    f"{starter.name} is {'sharper' if swing < 0 else 'struggling'} "
                    "of late"
                ),
                detail=(
                    f"{recent_era:.2f} ERA over his last "
                    f"{recent_stats.get('games')} starts against "
                    f"{season_era:.2f} on the season"
                ),
                value=f"{swing:+.2f}",
                team=team.abbreviation,
            )
        )

    # -- bullpen availability ----------------------------------------------
    for team in teams:
        relievers = [
            p for p in team.pitchers
            if not p.is_probable_starter and p.role.startswith("RP")
        ]
        if not relievers:
            continue
        rested = [
            p for p in relievers
            if p.availability and (p.availability.days_rest or 0) >= 2
        ]
        if len(rested) <= 2:
            out.append(
                Highlight(
                    kind="cold",
                    headline=f"{team.abbreviation}'s bullpen is short today",
                    detail=(
                        f"only {len(rested)} of {len(relievers)} relievers have "
                        "two or more days of rest"
                    ),
                    value=f"{len(rested)}/{len(relievers)}",
                    team=team.abbreviation,
                )
            )

    # -- widest team-rank mismatch -----------------------------------------
    home_profile = getattr(bundle, "home_profile", None)
    away_profile = getattr(bundle, "away_profile", None)
    if home_profile and away_profile:
        gaps: list[tuple[int, str, str, Any, Any]] = []
        for side, other in ((home_profile, away_profile), (away_profile, home_profile)):
            for group in ("hitting", "pitching"):
                for mine, theirs in zip(
                    getattr(side, group), getattr(other, group)
                ):
                    if mine.rank is None or theirs.rank is None:
                        continue
                    gaps.append(
                        (theirs.rank - mine.rank, group, mine.label, side, mine)
                    )
        if gaps:
            gaps.sort(key=lambda item: item[0], reverse=True)
            gap, group, label, side, stat = gaps[0]
            out.append(
                Highlight(
                    kind="edge",
                    headline=f"{side.abbreviation} holds the widest edge in {label}",
                    detail=(
                        f"ranked {stat.rank} of 30 in {group}, "
                        f"{gap} places clear of their opponent"
                    ),
                    value=f"#{stat.rank}",
                    team=side.abbreviation,
                )
            )

    return out
