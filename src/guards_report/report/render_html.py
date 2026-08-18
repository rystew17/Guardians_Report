"""Render a ReportBundle to a single self-contained HTML file.

The renderer formats; it never computes. Every number it displays was produced
by metrics/formulas.py before it got here. The formatting filters below are
deliberately strict about one thing: a value of None means "not computable",
and it renders as a dash, never as 0.000.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader, select_autoescape

from guards_report.ingest.preview import ReportBundle

TEMPLATE_DIR = Path(__file__).parent / "templates"

# A dash, not a zero. The distinction matters: a hitter with no plate
# appearances against left-handers has an undefined average, not a .000 one.
EMPTY = "–"


def rate3(value: Any) -> str:
    """Format a rate as MLB does: .305 rather than 0.305."""
    if value is None:
        return EMPTY
    text = f"{float(value):.3f}"
    return text[1:] if text.startswith("0.") else text


def rate2(value: Any) -> str:
    if value is None:
        return EMPTY
    return f"{float(value):.2f}"


def pct1(value: Any, *, already_pct: bool = False) -> str:
    """Format a proportion as a percentage.

    `already_pct` distinguishes our own rates (0-1, from formulas.py) from
    Savant's, which arrive pre-multiplied (0-100). Conflating them would show
    a 27% whiff rate as 2700%.
    """
    if value is None:
        return EMPTY
    number = float(value)
    if not already_pct:
        number *= 100.0
    return f"{number:.1f}%"


def integer(value: Any) -> str:
    if value is None:
        return EMPTY
    return f"{int(value):,}"


def innings(value: Any) -> str:
    """Innings in MLB notation, where .1 and .2 mean one and two outs."""
    if value is None:
        return EMPTY
    return f"{float(value):.1f}"


def percentile_class(value: Any) -> str:
    """Bucket a 0-100 percentile for colour coding."""
    if value is None or value == "":
        return "pct-none"
    number = float(value)
    if number >= 75:
        return "pct-great"
    if number >= 60:
        return "pct-good"
    if number >= 40:
        return "pct-avg"
    if number >= 25:
        return "pct-poor"
    return "pct-bad"


def build_environment() -> Environment:
    env = Environment(
        loader=FileSystemLoader(TEMPLATE_DIR),
        autoescape=select_autoescape(["html"]),
        trim_blocks=True,
        lstrip_blocks=True,
    )
    env.filters.update(
        rate3=rate3,
        rate2=rate2,
        pct1=pct1,
        integer=integer,
        innings=innings,
        percentile_class=percentile_class,
    )
    return env


def render(bundle: ReportBundle, *, output_dir: Path) -> Path:
    env = build_environment()
    template = env.get_template("report.html")

    html = template.render(
        bundle=bundle,
        guardians=bundle.guardians,
        opponent=bundle.opponent,
        generated=bundle.generated_at.strftime("%Y-%m-%d %H:%M UTC"),
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    matchup = f"{bundle.away.abbreviation}-at-{bundle.home.abbreviation}"
    path = output_dir / f"{bundle.game_date.isoformat()}_{matchup}.html"
    path.write_text(html, encoding="utf-8")
    return path
