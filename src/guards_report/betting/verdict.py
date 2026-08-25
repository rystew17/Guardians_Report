"""Bet or don't, and the reason in plain words.

Every row on the page ends in a decision, and a decision without its reason is
worth nothing to a reader who has to judge whether to trust it. Four outcomes,
and three of them are "no":

**No record.** The model behind this market has never been checked against
outcomes, so there is no honest basis for a stake. This is not a technicality:
sizing a bet requires knowing how wrong we tend to be, and for totals, first
five and strikeout props we do not know.

**Priced in.** Our number is behind what the price demands. Note the bar is the
price, not the book's opinion -- at -135 the break-even is 57.4% while the
book's own belief is nearer 55%, and those 2.4 points are the margin. Agreeing
with the book is a losing position held with conviction.

**Inside our own error.** Positive expected value, but the edge is small
relative to how uncertain we are about our own number. This is the interesting
refusal, and it is the one a reader will most want to argue with -- so it says
how big the edge is and how big the uncertainty is, and lets them.

**Bet.** Clear of the price and clear of our own noise, with a stake attached.

The wording avoids certainty on purpose. Nothing here has been validated
against a closing-line record yet, and until it has, these are disagreements
with a bookmaker rather than known edges.
"""

from __future__ import annotations

from dataclasses import dataclass

from guards_report.betting.edge import Play

# Shown verbatim at the top of the page and in the header of every table.
DISCLAIMER = "Not Gambling Advice"

BET = "bet"
PASS_PRICED_IN = "priced-in"
PASS_INSIDE_ERROR = "inside-error"
PASS_UNMEASURED = "unmeasured"
PASS_DUPLICATE = "duplicate"


@dataclass(frozen=True)
class Verdict:
    """One decision, with the sentence that justifies it."""

    action: str          # BET | PASS_*
    label: str           # what the cell says
    reason: str          # one sentence, plain language
    detail: str          # the numbers behind the sentence

    @property
    def is_bet(self) -> bool:
        return self.action == BET


def _opening(text: str) -> str:
    """Capitalize a fragment that has become the start of a sentence.

    The basis strings read naturally mid-sentence ("from the starter strikeout
    distribution") and badly at the front of one. `.capitalize()` is wrong here
    -- it lowercases everything after the first letter, which would turn "Model
    A" into "Model a".
    """
    return (text[:1].upper() + text[1:]) if text else ""


def _pts(value: float) -> str:
    """Probability differences read as points, which is how they are argued."""
    return f"{value * 100:+.1f} pt"


def decide(
    play: Play,
    *,
    measured: bool,
    z_threshold: float,
    basis: str = "",
    superseded_by: str = "",
) -> Verdict:
    """The verdict for one priced selection."""
    if superseded_by:
        return Verdict(
            action=PASS_DUPLICATE,
            label="Covered",
            reason=(
                "The same position is already staked at a better price. Taking "
                "both would be one wager at double stake wearing two names."
            ),
            detail=(
                f"{superseded_by} carries this bet. These win and lose together, "
                f"so the stake belongs on one of them."
            ),
        )
    if not measured:
        return Verdict(
            action=PASS_UNMEASURED,
            label="No",
            reason=(
                "We have never checked this model against outcomes, so we "
                "cannot say how wrong it usually is — and a stake is a claim "
                "about exactly that."
            ),
            detail=(
                f"{_opening(basis) or 'This model'} has no calibration record. "
                f"Our number is {play.p_model:.1%} against a price demanding "
                f"{play.break_even:.1%}."
            ),
        )

    if play.expected_value <= 0:
        shortfall = play.break_even - play.p_used
        return Verdict(
            action=PASS_PRICED_IN,
            label="No",
            reason=(
                "The price already asks for more than we think is there. "
                "Being right is not what pays; the price being wrong is."
            ),
            detail=(
                f"Break-even is {play.break_even:.1%} and we make it "
                f"{play.p_used:.1%} — short by {shortfall * 100:.1f} points. "
                f"The book's own belief is {play.p_market:.1%}; the gap between "
                f"that and break-even is the {(play.break_even - play.p_market) * 100:.1f} "
                f"point margin."
            ),
        )

    if play.z < z_threshold:
        return Verdict(
            action=PASS_INSIDE_ERROR,
            label="Not yet",
            reason=(
                "There is an edge here, but it is smaller than our own "
                "uncertainty about the number that produced it."
            ),
            detail=(
                f"Edge of {_pts(play.edge)} against a standard error of "
                f"{play.posterior_sd * 100:.1f} points — z of {play.z:.2f}, "
                f"short of the {z_threshold:.1f} bar. We would need our "
                f"probability to be {_pts(play.p_model - play.p_market)} clear "
                f"of the market rather than what it is."
            ),
        )

    return Verdict(
        action=BET,
        label="Bet",
        reason=(
            "Our number clears both the price and our own margin for error."
        ),
        detail=(
            f"We make it {play.p_used:.1%} against a break-even of "
            f"{play.break_even:.1%} — an edge of {_pts(play.edge)}, or "
            f"{play.z:.1f} times our standard error. Stake "
            f"{play.stake * 100:.1f}% of bankroll at quarter Kelly."
        ),
    )


def summarize(verdicts: list[Verdict]) -> str:
    """One line for the top of the section.

    A night with nothing to bet is the expected outcome and has to read as the
    system working rather than as a page that failed to load.
    """
    if not verdicts:
        return "No prices entered for tonight."

    bets = [v for v in verdicts if v.is_bet]
    if bets:
        plural = "" if len(bets) == 1 else "s"
        return (
            f"{len(bets)} play{plural} from {len(verdicts)} priced selections."
        )

    unmeasured = sum(1 for v in verdicts if v.action == PASS_UNMEASURED)
    close = sum(1 for v in verdicts if v.action == PASS_INSIDE_ERROR)
    if close:
        return (
            f"Nothing to bet. {close} selection{'' if close == 1 else 's'} showed "
            f"an edge too small to separate from our own error."
        )
    if unmeasured == len(verdicts):
        return (
            "Nothing to bet. Every market priced tonight comes from a model we "
            "have not yet checked against outcomes."
        )
    if unmeasured:
        priced_in = len(verdicts) - unmeasured
        return (
            f"Nothing to bet. {priced_in} selection"
            f"{'' if priced_in == 1 else 's'} already priced in, and "
            f"{unmeasured} from models we have not checked against outcomes."
        )
    return "Nothing to bet. Every price tonight already accounts for what we know."
