"""Lineup-aware team offence.

The single largest gap between this model and a book's is that a book knows who
is playing. A club missing its three best hitters is a different club, and Elo
cannot know that until after the games have been played -- which is precisely
the flat middle where 69% of games sit and where the win model currently
contributes almost nothing over a coin flip.

MLB posts official lineups roughly three hours before first pitch, so a report
generated inside that window can use the real card. One generated the night
before cannot. Both have to work, and they must not be silently interchangeable:
a projection built on nine named hitters and one built on a guess are different
claims, and the report should say which it is making.

**Historical lineups come free.** The plate-appearance corpus records who batted
and in what order, so the training set needs no separate boxscore fetch. The
ordering comes from Statcast's own `at_bat_number`; reconstructing it from row
order instead disagreed with the official boxscore on four of four games tested,
while `at_bat_number` matched on six of six.

**Only the starting nine count.** Taking everyone who batted would fold in
pinch-hitters and defensive replacements -- players whose appearance is a
*consequence* of how the game went. A model trained on that would learn from the
future and then find nothing like it at prediction time, when only the posted
card exists.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

# Plate appearances a lineup slot gets in a typical nine-inning game. The
# leadoff spot bats roughly three-quarters of a time more than the ninth, so
# weighting slots equally would misstate a lineup's value -- and would miss the
# thing managers are actually doing when they move a hot hitter up.
# Measured from the corpus by `slot_weights`; these are the fallback.
DEFAULT_SLOT_PA = (4.65, 4.55, 4.44, 4.34, 4.23, 4.12, 4.02, 3.92, 3.82)

# A lineup this incomplete is not a lineup. Below it, fall back rather than
# pretend a partial card describes the team.
MIN_SLOTS = 7


@dataclass
class LineupValue:
    """A lineup's expected offensive value, and how much was actually known."""

    expected_value: float = 0.0
    slots_known: int = 0
    source: str = "none"
    thin: list[int] = field(default_factory=list)
    batters: list[int] = field(default_factory=list)

    @property
    def is_official(self) -> bool:
        return self.source == "posted"


def slot_weights(pa_corpus: pd.DataFrame) -> tuple[float, ...]:
    """Plate appearances per lineup slot, measured rather than assumed.

    Measure on the era being modelled. Before the universal designated hitter
    arrived in 2022, the ninth slot in a National League park held a pitcher who
    was pinch-hit for early, and it shows: measured on 2016 the ninth spot takes
    2.87 plate appearances against 3.68 for the eighth, a cliff that does not
    exist afterwards. One weight vector spanning 2015 to 2026 would describe
    neither era.
    """
    starters = starting_lineups(pa_corpus)
    if starters.empty:
        return DEFAULT_SLOT_PA

    counts = (
        pa_corpus.merge(
            starters[["game_pk", "batting_team", "batter", "slot"]],
            on=["game_pk", "batting_team", "batter"], how="inner",
        )
        .groupby(["game_pk", "batting_team", "slot"])
        .size()
        .groupby("slot").mean()
    )
    return tuple(float(counts.get(i, DEFAULT_SLOT_PA[i - 1])) for i in range(1, 10))


def slot_weights_by_season(pa_corpus: pd.DataFrame) -> dict[int, tuple[float, ...]]:
    """One weight vector per season, so the designated-hitter change is followed.

    Keyed by season rather than by a hard-coded 2022 boundary: the rule changed
    once, but measuring each year separately means the code follows the data
    instead of encoding a date that a future rule change would falsify.
    """
    return {
        int(season): slot_weights(frame)
        for season, frame in pa_corpus.groupby("season", sort=True)
    }


def starting_lineups(pa_corpus: pd.DataFrame) -> pd.DataFrame:
    """The nine who started, per team-game, with their batting slot.

    The first nine distinct batters in `at_bat_number` order are exactly the
    posted card: a substitute cannot appear before every starter has hit once,
    because the order does not skip.
    """
    if pa_corpus.empty:
        return pd.DataFrame(columns=["game_pk", "batting_team", "batter", "slot"])

    ordered = pa_corpus.sort_values(["game_pk", "batting_team", "at_bat_number"])
    first = (
        ordered.drop_duplicates(["game_pk", "batting_team", "batter"])
        .groupby(["game_pk", "batting_team"], sort=False)
        .head(9)
        .copy()
    )
    first["slot"] = first.groupby(["game_pk", "batting_team"], sort=False).cumcount() + 1
    return first[["game_pk", "batting_team", "batter", "slot", "game_date", "season"]]


def value_of(
    batters: list[int],
    talent: Any,
    *,
    pitcher_id: int | None = None,
    pitcher_throws: str = "R",
    stands: dict[int, str] | None = None,
    weights: tuple[float, ...] = DEFAULT_SLOT_PA,
    source: str = "posted",
    thin_pa: int = 50,
) -> LineupValue:
    """Expected run value per plate appearance for this lineup.

    Weighted by how often each slot bats, and -- when the opposing starter is
    known -- evaluated against that specific pitcher's hand, which is where the
    platoon advantage enters. A lineup stacked with left-handed hitters is worth
    materially more against a right-hander than the same nine against a lefty,
    and a season-level team wOBA cannot express that.
    """
    slots = [int(b) for b in batters[:9] if b is not None]
    if len(slots) < MIN_SLOTS:
        return LineupValue(source="none", slots_known=len(slots))

    stands = stands or {}
    total_weight, total_value, thin = 0.0, 0.0, []

    for index, batter in enumerate(slots):
        weight = weights[index] if index < len(weights) else weights[-1]
        stand = stands.get(batter, "R")
        if pitcher_id is not None:
            value = talent.expected_value(
                batter, pitcher_id, stand=stand, throws=pitcher_throws
            )
        else:
            value = talent.intercept + (talent.batter_score(batter) or 0.0)
        total_value += weight * value
        total_weight += weight
        if talent.evidence(batter, side="batter") < thin_pa:
            thin.append(batter)

    return LineupValue(
        expected_value=total_value / total_weight if total_weight else 0.0,
        slots_known=len(slots),
        source=source,
        thin=thin,
        batters=slots,
    )


def recent_regulars(
    pa_corpus: pd.DataFrame, team: str, *, before, games: int = 10
) -> list[int]:
    """Best guess at a lineup when none has been posted.

    The most frequent starter in each slot over the club's last few games. This
    is a guess and is labeled as one -- it is right about the shape of a lineup
    and wrong about exactly who is resting today, which is the very thing the
    posted card would tell us.
    """
    prior = pa_corpus[
        (pa_corpus["batting_team"] == team) & (pa_corpus["game_date"] < before)
    ]
    if prior.empty:
        return []

    recent_games = sorted(prior["game_pk"].unique())[-games:]
    starters = starting_lineups(prior[prior["game_pk"].isin(recent_games)])
    if starters.empty:
        return []

    chosen, used = [], set()
    for slot in range(1, 10):
        counts = starters[starters["slot"] == slot]["batter"].value_counts()
        for batter, _ in counts.items():
            if batter not in used:
                chosen.append(int(batter))
                used.add(batter)
                break
    return chosen


def build_feature(
    games: pd.DataFrame,
    pa_corpus: pd.DataFrame,
    talent_by_season: dict[int, Any],
    *,
    weights: tuple[float, ...] | None = None,
) -> pd.DataFrame:
    """Per-game lineup value for both clubs, for model fitting.

    Talent is taken from the fit trained on seasons strictly before this one, so
    a projection never benefits from knowing how the hitters ended up doing.
    """
    # Weights vary by era, so they are looked up per season unless a caller
    # deliberately pins one vector.
    pinned = weights
    by_season = {} if pinned else slot_weights_by_season(pa_corpus)
    starters = starting_lineups(pa_corpus)
    hands = (
        pa_corpus.drop_duplicates("batter").set_index("batter")["stand"].to_dict()
    )
    opposing = (
        pa_corpus.drop_duplicates(["game_pk", "batting_team"])
        .set_index(["game_pk", "batting_team"])["p_throws"].to_dict()
    )

    grouped = {
        key: list(frame.sort_values("slot")["batter"])
        for key, frame in starters.groupby(["game_pk", "batting_team"], sort=False)
    }

    rows = []
    for game in games.itertuples(index=False):
        season = int(game.season)
        talent = talent_by_season.get(season)
        if talent is None:
            continue
        record: dict[str, Any] = {"game_pk": game.game_pk}
        for side, team, opp_starter in (
            ("home", game.home_team, game.away_starter_id),
            ("away", game.away_team, game.home_starter_id),
        ):
            batters = grouped.get((game.game_pk, team), [])
            value = value_of(
                batters, talent,
                pitcher_id=int(opp_starter) if opp_starter == opp_starter else None,
                pitcher_throws=opposing.get((game.game_pk, team), "R") or "R",
                stands=hands,
                weights=pinned or by_season.get(season, DEFAULT_SLOT_PA),
            )
            record[f"{side}_lineup_value"] = value.expected_value
            record[f"{side}_lineup_slots"] = value.slots_known
        rows.append(record)

    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    frame["lineup_diff"] = frame["home_lineup_value"] - frame["away_lineup_value"]
    return frame
