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

from jinja2 import Environment, FileSystemLoader, Undefined, select_autoescape
from markupsafe import Markup

from guards_report.config import SPLIT_LABELS
from guards_report.ingest.preview import ReportBundle
from guards_report.metrics.trends import reliability as _reliability
from guards_report.metrics.zones import ZoneCell, ZoneGrid

TEMPLATE_DIR = Path(__file__).parent / "templates"

EMPTY = "–"


def _missing(value: Any) -> bool:
    """True when a value should render as a dash.

    Covers three cases that all mean "we do not have this": an explicit None
    from a formula whose denominator was zero, a Jinja Undefined from a lookup
    on a player who has no row in some leaderboard, and an empty string from a
    blank CSV cell. All three must render as a dash rather than a zero.
    """
    return value is None or isinstance(value, Undefined) or value == ""


def rate3(value: Any) -> str:
    """Format a rate as MLB does: .305 rather than 0.305."""
    if _missing(value):
        return EMPTY
    text = f"{float(value):.3f}"
    return text[1:] if text.startswith("0.") else text


def rate2(value: Any) -> str:
    if _missing(value):
        return EMPTY
    return f"{float(value):.2f}"


def pct1(value: Any, *, already_pct: bool = False) -> str:
    """Format a proportion as a percentage.

    `already_pct` distinguishes our own rates (0-1, from formulas.py) from
    Savant's, which arrive pre-multiplied (0-100). Conflating them would show a
    27% whiff rate as 2700%.
    """
    if _missing(value):
        return EMPTY
    number = float(value)
    if not already_pct:
        number *= 100.0
    return f"{number:.1f}%"


def integer(value: Any) -> str:
    if _missing(value):
        return EMPTY
    return f"{int(value):,}"


def innings(value: Any) -> str:
    if _missing(value):
        return EMPTY
    return f"{float(value):.1f}"


def percentile_class(value: Any) -> str:
    if _missing(value):
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

    Prefers the colour MLB supplies with the zone, so the map matches what the
    same player looks like on MLB and Savant and the shading stays a sourced
    value. Falls back to a blue-to-red scale over the player's own range only
    when the source omits a colour.
    """
    if cell.color:
        # MLB sends these at .55 alpha for overlay on a white field; opaque
        # here since we paint them directly onto the cell.
        return f"background: {_opaque(cell.color)};"

    if cell.value is None:
        return "background: var(--zone-empty);"

    intensity = grid.intensity(cell)
    if intensity >= 0.5:
        weight = (intensity - 0.5) * 2
        return f"background: rgba(210, 45, 73, {0.12 + weight * 0.68:.2f});"
    weight = (0.5 - intensity) * 2
    return f"background: rgba(50, 90, 168, {0.12 + weight * 0.68:.2f});"


def _opaque(color: str) -> str:
    """Raise an rgba() colour's alpha so it reads clearly as a filled cell."""
    text = color.strip()
    if not text.startswith("rgba"):
        return text
    inside = text[text.find("(") + 1 : text.rfind(")")]
    parts = [p.strip() for p in inside.split(",")]
    if len(parts) != 4:
        return text
    return f"rgba({parts[0]}, {parts[1]}, {parts[2]}, 0.92)"


def sparkline(series: Any, *, width: int = 132, height: int = 26) -> Markup:
    """Inline SVG sparkline for a rolling trend series.

    Scaled to the series' own range rather than an absolute one: the question
    is whether this player is trending up or down, and a fixed scale flattens
    that for anyone whose numbers sit in a narrow band. The final point is
    emphasised because "where is he now" is what the reader is looking for.
    """
    if series is None or not getattr(series, "has_data", False):
        return Markup("")

    points = series.points
    low, high = min(points), max(points)
    span = high - low or 1.0
    step = width / max(1, len(points) - 1)
    pad = 3

    coords = [
        (index * step, pad + (height - 2 * pad) * (1 - (value - low) / span))
        for index, value in enumerate(points)
    ]
    path = " ".join(
        f"{'M' if i == 0 else 'L'}{x:.1f},{y:.1f}" for i, (x, y) in enumerate(coords)
    )
    area = (
        f"M0,{height} L" + " L".join(f"{x:.1f},{y:.1f}" for x, y in coords)
        + f" L{coords[-1][0]:.1f},{height} Z"
    )
    end_x, end_y = coords[-1]
    stroke = {"up": "var(--good)", "down": "var(--bad)"}.get(
        series.direction, "var(--slate)"
    )

    return Markup(
        f'<svg class="spark" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" role="img" '
        f'aria-label="{series.label}, trending {series.direction}">'
        f'<path d="{area}" fill="{stroke}" opacity=".10"/>'
        f'<path d="{path}" fill="none" stroke="{stroke}" stroke-width="1.6" '
        f'stroke-linejoin="round" stroke-linecap="round"/>'
        f'<circle cx="{end_x:.1f}" cy="{end_y:.1f}" r="2.4" fill="{stroke}"/>'
        f"</svg>"
    )


def trend_arrow(series: Any) -> Markup:
    if series is None or not getattr(series, "has_data", False):
        return Markup("")
    glyph = {"up": "▲", "down": "▼"}.get(series.direction, "—")
    return Markup(f'<span class="tr tr-{series.direction}">{glyph}</span>')


def reliability_class(entry: Any) -> str:
    """Class marking how much weight a rate deserves given its denominator."""
    if _missing(entry):
        return ""
    return f"rel-{entry.band}"


def rank_ordinal(rank: Any) -> str:
    """Render a 1-30 league rank as 1st, 2nd, 3rd and so on."""
    if _missing(rank):
        return EMPTY
    number = int(rank)
    if 11 <= number % 100 <= 13:
        suffix = "th"
    else:
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(number % 10, "th")
    return f"{number}{suffix}"


def statcast_zone_style(chart: Any, code: str) -> str:
    """Shading for a pitch-level zone cell, on a blue-to-red diverging scale.

    Scaled to the player's own range so the map answers "where is he strong
    relative to himself". Cells with almost no sample behind them stay neutral
    rather than being painted a confident colour on three plate appearances.

    Swing and strikeout rates are inverted: a high chase rate is a weakness,
    and colouring it red would tell the reader the opposite of the truth.
    """
    value = chart.value(code)
    if value is None or chart.sample(code) < 4:
        return "background: var(--zone-empty); color: var(--faint);"

    intensity = chart.intensity(code)
    if chart.metric in ("k", "whiff", "swing"):
        intensity = 1.0 - intensity

    if intensity >= 0.5:
        weight = (intensity - 0.5) * 2
        return f"background: rgba(210, 45, 73, {0.10 + weight * 0.72:.2f});"
    weight = (0.5 - intensity) * 2
    return f"background: rgba(50, 90, 168, {0.10 + weight * 0.72:.2f});"


def zone_value(chart: Any, code: str) -> str:
    """Format a zone cell according to its metric.

    Rate metrics carry one decimal like every other percentage in the report:
    the difference between a 28% and a 28.4% chase rate in a given zone is the
    kind of thing this map exists to show, and rounding it away costs the
    resolution the reader came for.
    """
    value = chart.value(code)
    if value is None or chart.sample(code) < 4:
        return EMPTY
    if chart.metric in ("swing", "k", "whiff"):
        return f"{value * 100:.1f}%"
    return rate3(value)


# Outcome colours for the spray chart. Outs stay quiet so hits carry the eye.
SPRAY_COLORS = {
    "out": ("#9aa4af", 2.3),
    "single": ("#2f7d4f", 3.4),
    "double": ("#1f6fb8", 3.9),
    "triple": ("#8b45b5", 4.4),
    "home_run": ("#d22d49", 5.0),
}


def spray_chart(chart: Any, *, width: int = 290, height: int = 258) -> Markup:
    """Inline SVG spray chart: batted balls plotted on a field outline.

    Left-handed hitters are mirrored upstream so pull is always the same side
    of the image, which is what lets two hitters be compared at a glance.
    """
    if chart is None or not getattr(chart, "has_data", False):
        return Markup("")

    max_feet = 430.0
    cx, cy = width / 2, height - 16
    scale = min((width / 2 - 6) / (max_feet * 0.72), (height - 26) / max_feet)

    def project(x: float, y: float) -> tuple[float, float]:
        return cx + x * scale, cy - y * scale

    corner = 0.7071
    lf = project(-330 * corner, 330 * corner)
    rf = project(330 * corner, 330 * corner)
    arc_r = 405 * scale
    b2 = project(0, 127.3)
    b1 = project(90 * corner, 90 * corner)
    b3 = project(-90 * corner, 90 * corner)

    parts = [
        f'<svg class="spray" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" role="img" '
        f'aria-label="Spray chart of {len(chart.points)} batted balls">',
        f'<path d="M{cx:.1f},{cy:.1f} L{lf[0]:.1f},{lf[1]:.1f} '
        f'A{arc_r:.1f},{arc_r:.1f} 0 0,1 {rf[0]:.1f},{rf[1]:.1f} Z" '
        f'fill="var(--field)" stroke="var(--field-line)" stroke-width="1"/>',
        f'<path d="M{cx:.1f},{cy:.1f} L{b3[0]:.1f},{b3[1]:.1f} '
        f'L{b2[0]:.1f},{b2[1]:.1f} L{b1[0]:.1f},{b1[1]:.1f} Z" '
        f'fill="none" stroke="var(--field-line)" stroke-width="1"/>',
    ]

    # Outs first, so hits are never buried underneath them.
    ordered = sorted(chart.points, key=lambda p: list(SPRAY_COLORS).index(p.outcome))
    for point in ordered:
        color, radius = SPRAY_COLORS.get(point.outcome, ("#9aa4af", 2.3))
        px, py = project(point.x, point.y)
        if not (-4 <= px <= width + 4 and -4 <= py <= height + 4):
            continue
        opacity = "0.45" if point.outcome == "out" else "0.9"
        parts.append(
            f'<circle cx="{px:.1f}" cy="{py:.1f}" r="{radius}" fill="{color}" '
            f'opacity="{opacity}"/>'
        )

    parts.append("</svg>")
    return Markup("".join(parts))


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
        sparkline=sparkline,
        trend_arrow=trend_arrow,
        rank_ordinal=rank_ordinal,
        reliability_class=reliability_class,
        spray_chart=spray_chart,
    )
    env.globals.update(
        zone_cell_style=zone_cell_style,
        statcast_zone_style=statcast_zone_style,
        zone_value=zone_value,
        reliability=_reliability,
        split_labels=SPLIT_LABELS,
        spray_colors=SPRAY_COLORS,
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
