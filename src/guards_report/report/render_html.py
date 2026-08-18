"""Render a ReportBundle to a single self-contained HTML file.

The renderer formats; it never computes. Every number it displays was produced
by metrics/ before it got here. It is strict about one thing: a value of None
means "not computable", and it renders as a dash, never as 0.000 -- a hitter
with no plate appearances against left-handers has an undefined average, not a
.000 one.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader, select_autoescape
from markupsafe import Markup

from guards_report.ingest.preview import ReportBundle
from guards_report.metrics.zones import ZoneCell, ZoneGrid

TEMPLATE_DIR = Path(__file__).parent / "templates"

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
    Savant's, which arrive pre-multiplied (0-100). Conflating them would show a
    27% whiff rate as 2700%.
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
    if value is None:
        return EMPTY
    return f"{float(value):.1f}"


def percentile_class(value: Any) -> str:
    if value is None or value == "":
        return "pct-none"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "pct-none"
    if number >= 75:
        return "pct-great"
    if number >= 60:
        return "pct-good"
    if number >= 40:
        return "pct-avg"
    if number >= 25:
        return "pct-poor"
    return "pct-bad"


def delta_html(entry: dict[str, Any] | None, *, style: str = "rate3") -> Markup:
    """Render a league-average delta as a small signed, coloured annotation.

    Direction is decided in metrics/league_averages.py, which knows that a low
    ERA is good and a low OPS is not. The renderer only paints it.
    """
    if not entry:
        return Markup("")

    difference = entry["diff"]
    if style == "pct":
        text = f"{difference * 100:+.1f}"
    elif style == "rate2":
        text = f"{difference:+.2f}"
    else:
        text = f"{difference:+.3f}".replace("+0.", "+.").replace("-0.", "-.")

    return Markup(
        f'<span class="d d-{entry["direction"]}">{text}</span>'
    )


def zone_cell_style(grid: ZoneGrid, cell: ZoneCell) -> str:
    """Background shading for one heat-map cell.

    Shaded on a blue-to-red scale against the player's own range, which is what
    makes a heat map answer "where is this hitter strong relative to himself"
    rather than washing out for anyone uniformly good or uniformly bad.
    """
    if cell.value is None:
        return "background: var(--zone-empty);"

    intensity = grid.intensity(cell)
    if intensity >= 0.5:
        weight = (intensity - 0.5) * 2
        return f"background: rgba(210, 45, 73, {0.12 + weight * 0.68:.2f});"
    weight = (0.5 - intensity) * 2
    return f"background: rgba(50, 90, 168, {0.12 + weight * 0.68:.2f});"


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
        delta=delta_html,
    )
    env.globals.update(zone_cell_style=zone_cell_style)
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
