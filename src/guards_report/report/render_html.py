"""Render a ReportBundle to a single self-contained HTML file.

The renderer formats; it never computes. Every number it displays was produced
by metrics/ before it got here. It is strict about one thing: a value of None
means "not computable", and it renders as a dash, never as 0.000 -- a hitter
with no plate appearances against left-handers has an undefined average, not a
.000 one.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader, Undefined, select_autoescape
from markupsafe import Markup

from guards_report.config import SPLIT_LABELS
from guards_report.ingest.preview import ReportBundle
from guards_report.report import preview_card, share_page
from guards_report.metrics import clocks, series
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

    Prefers the color MLB supplies with the zone, so the map matches what the
    same player looks like on MLB and Savant and the shading stays a sourced
    value. Falls back to a blue-to-red scale over the player's own range only
    when the source omits a color.
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
    """Raise an rgba() color's alpha so it reads clearly as a filled cell."""
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
        series.direction, "var(--muted)"
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
    """Render an integer rank or percentile as 1st, 2nd, 3rd and so on."""
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
    rather than being painted a confident color on three plate appearances.

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


# Outcome colors for the spray chart. Outs stay quiet so hits carry the eye.
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
        confidence_strip=confidence_strip,
        waterfall_svg=waterfall_svg,
        margin_svg=margin_svg,
        runs_by_side_svg=runs_by_side_svg,
        totals_svg=totals_svg,
        rate_compare=rate_compare,
        elo_scale_svg=elo_scale_svg,
        accuracy_bar=accuracy_bar,
        three_way_bar=three_way_bar,
        strikeout_distribution=strikeout_distribution,
        prop_bar=prop_bar,
        hit_spread=hit_spread,
        projection_read=_projection_read,
    )
    return env


def _projection_read(projection: Any) -> dict:
    """The sentence the decomposition supports, selected by arithmetic.

    Lives here only so the template can reach it; every branch is a threshold on
    a measured contribution, and no text is model-generated.
    """
    from guards_report.projections.predict import read_of

    return read_of(projection) if projection is not None else {}


def _add_table_semantics(html: str) -> str:
    """Give every header cell a scope, so a screen reader can pair value to column.

    Applied to the rendered document rather than to each template: there are
    hundreds of tables across the player pages, and a rule that has to be
    remembered at every one of them is a rule that will be missed.

    A `<th>` inside `<thead>` labels a column; one elsewhere labels its row.
    """
    # `<th` is a prefix of `<thead`, so a pattern that does not require a
    # delimiter after it rewrites every opening `<thead>` into
    # `<th scope="col"ead>`. Browsers recover from that by closing the malformed
    # cell and starting the table with a stray empty header row -- which renders
    # as a thin band above the real header and is easy to read as a style
    # choice. It had corrupted all 218 table headers in the document, and the
    # only visible trace was that band.
    #
    # The lookahead for whitespace or `>` is the whole fix; the negative
    # lookahead below only avoids double-scoping a cell that already has one.
    html = re.sub(r"<th(?=[\s>])(?![^>]*scope=)", '<th scope="col"', html)
    return html


def render(bundle: ReportBundle, *, output_dir: Path, bucket: str = "") -> Path:
    env = build_environment()
    template = env.get_template("report.html")

    html = template.render(
        bundle=bundle,
        # Open Graph, so a pasted link unfurls into something that names the
        # game rather than a bare storage path.
        preview_title=preview_title(bundle),
        preview_description=preview_description(bundle),
        preview_url=preview_url(bundle, bucket=bucket),
        preview_image=preview_image_url(bundle, bucket=bucket),
        guardians=bundle.guardians,
        opponent=bundle.opponent,
        generated=clocks.stamp(bundle.generated_at),
        # Exposed to the template so first-pitch and audit timestamps are
        # formatted by the same code rather than by ad-hoc strftime calls.
        game_time=clocks.game_time,
        stamp=clocks.stamp,
        # The ranking formulas are printed in the report so "top performers" is
        # checkable rather than asserted.
        batter_score_formula=series.BATTER_SCORE,
        pitcher_score_formula=series.PITCHER_SCORE,
    )

    html = _add_table_semantics(html)

    # The card is drawn beside the report so `publish` can upload the pair. X
    # renders a link card image-first and shows a bare URL without one, which
    # is the whole reason this exists.
    try:
        preview_card.build(bundle, output_dir=output_dir)
        # And a two-kilobyte page carrying the same tags, for crawlers that
        # will not open a 2.7 MB document.
        share_page.build(
            bundle, output_dir=output_dir,
            report_url=preview_url(bundle, bucket=bucket),
            image_url=preview_image_url(bundle, bucket=bucket),
            title=preview_title(bundle),
            description=preview_description(bundle),
        )
    except Exception as exc:  # noqa: BLE001 -- a missing asset is not a failure
        print(f"  warning: preview assets not built ({exc})", file=sys.stderr)

    output_dir.mkdir(parents=True, exist_ok=True)
    matchup = f"{bundle.away.abbreviation}-at-{bundle.home.abbreviation}"
    path = output_dir / f"{bundle.game_date.isoformat()}_{matchup}.html"
    path.write_text(html, encoding="utf-8")
    return path

# ---------------------------------------------------------------------------
# Projection visuals
# ---------------------------------------------------------------------------
#
# Built to answer three questions the headline percentage cannot: how confident
# is this by the model's own standards, what pushed it there, and how does
# tonight differ from a normal night.
#
# That emphasis comes from a measurement. Across 25,192 games the model's
# expected-runs figure moves only 18% as much as real scoring does, and three
# rounded scorelines cover 70% of all games -- so a page built around the
# projected final would print nearly the same thing every day. The win
# probability does vary (0.23 to 0.83, sd 0.087) and is well calibrated in every
# bucket, but it lands between 0.45 and 0.65 on roughly six nights in ten. What
# genuinely changes night to night is which input is doing the work, so that is
# what these charts are built to show.

# Declared in the stylesheet so the projections share the report's palette
# rather than introducing a private one. See --home / --away / --mark.
HOME_INK = "var(--home)"
AWAY_INK = "var(--away)"
FLAG_INK = "var(--mark)"


def confidence_strip(projection: Any, *, width: int = 520, height: int = 118) -> Markup:
    """Tonight's conviction against every call this model has ever made.

    The common misreading of a win probability is treating 60% as a strong
    opinion. For this model it is a moderately strong one; for a different model
    it might be the most extreme output it has ever produced. The curve is this
    model's own sorted conviction across the corpus, so the marker's position
    answers "how sure is this, for something right 57% of the time" without the
    reader holding any of those figures in mind.
    """
    grid = (projection.reference or {}).get("win_prob_sorted") or []
    if not grid:
        return Markup("")

    # Folded around 0.5: the question is conviction, not which club is favoured.
    folded = sorted(max(v, 1.0 - v) for v in grid)
    lo, hi = folded[0], folded[-1]
    span = max(hi - lo, 1e-6)

    left, right, top, bottom = 34, 8, 14, 18
    plot_w, plot_h = width - left - right, height - top - bottom

    def px(i: int) -> float:
        return left + plot_w * i / max(len(folded) - 1, 1)

    def py(v: float) -> float:
        return top + plot_h * (1.0 - (v - lo) / span)

    line = " ".join(f"{px(i):.1f},{py(v):.1f}" for i, v in enumerate(folded))
    area = f"{left},{top + plot_h:.1f} {line} {left + plot_w},{top + plot_h:.1f}"

    here = max(0.0, min(100.0, projection.confidence_percentile))
    conviction = max(projection.win_probability, 1 - projection.win_probability)
    mx, my = left + plot_w * here / 100.0, py(conviction)

    parts = [
        f'<svg class="cstrip" '
        f'viewBox="0 0 {width} {height}" role="img" aria-label="Tonight at the '
        f'{here:.0f}th percentile of this model\'s confidence">',
        f'<polygon points="{area}" fill="var(--accent)" opacity="0.12"/>',
        f'<polyline points="{line}" fill="none" stroke="var(--accent)" '
        f'stroke-width="1.4" opacity="0.6"/>',
    ]
    for pct in (25, 50, 75):
        x = left + plot_w * pct / 100.0
        parts.append(
            f'<line x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{top + plot_h:.1f}" '
            f'stroke="var(--border)" stroke-width="0.6" stroke-dasharray="2 3"/>'
        )
    parts += [
        f'<line x1="{mx:.1f}" y1="{top}" x2="{mx:.1f}" y2="{top + plot_h:.1f}" '
        f'stroke="{FLAG_INK}" stroke-width="1.6"/>',
        f'<circle cx="{mx:.1f}" cy="{my:.1f}" r="4.2" fill="{FLAG_INK}"/>',
        f'<text class="sgl" x="{left - 6}" y="{py(hi) + 4:.1f}" text-anchor="end">'
        f'{hi * 100:.0f}%</text>',
        f'<text class="sgl" x="{left - 6}" y="{py(lo) + 4:.1f}" text-anchor="end">'
        f'{lo * 100:.0f}%</text>',
        f'<text class="sgl" x="{left}" y="{height - 5}">coin flip</text>',
        f'<text class="sgl" x="{left + plot_w}" y="{height - 5}" '
        f'text-anchor="end">most sure it gets</text>',
        "</svg>",
    ]
    return Markup("".join(parts))


def waterfall_svg(
    projection: Any, home: str, away: str, *, width: int = 600, row: int = 28
) -> Markup:
    """What moved this game off an average matchup, input by input.

    The model is linear on the log-odds scale, so this decomposition is exact
    rather than an attribution heuristic: the bars sum to the whole departure
    from a league-average game. It is also the part of the page that actually
    differs night to night. Two games can both read 60% while one is a gap in
    team quality and the other is an ordinary club with its best arm going --
    identical headline, opposite meaning.

    Bars stay in log-odds because that is the scale they add on. Converting each
    to percentage points separately would give numbers that do not sum to the
    total, which is exactly the misreading the chart should prevent.
    """
    from guards_report.projections.predict import FEATURE_LABELS

    rows = [
        r for r in (projection.contributions or [])
        if abs(r["contribution"]) > 1e-4
    ]
    if not rows:
        return Markup("")

    # Below a half-hundredth of a log-odd a bar cannot be seen; collect the tail
    # rather than drawing rows that only add height.
    major = [r for r in rows if abs(r["contribution"]) >= 0.005]
    minor = [r for r in rows if abs(r["contribution"]) < 0.005]
    if minor:
        major.append({
            "name": "_rest",
            "contribution": sum(r["contribution"] for r in minor),
            "imputed": False,
            "_label": f"{len(minor)} smaller inputs",
        })

    label_w, pad, gutter = 210, 8, 54
    values = [r["contribution"] for r in major]

    # A centered zero line is only worth its cost when the chart actually
    # diverges. When every input pushes the same way -- which is common, since
    # one club is usually better on most counts -- centring throws away half the
    # canvas and squeezes the bars into the remainder.
    diverges = min(values) < 0 < max(values)
    if diverges:
        axis = label_w + (width - label_w) / 2
        span = (width - label_w) / 2 - gutter
    elif max(values) > 0:
        axis = label_w
        span = width - label_w - gutter
    else:
        axis = width - gutter
        span = width - label_w - gutter

    peak = max(abs(v) for v in values) or 1.0
    height = pad + row * len(major) + 22

    parts = [
        f'<svg class="wfall" '
        f'viewBox="0 0 {width} {height}" role="img" '
        f'aria-label="What moved the win probability">'
    ]

    for i, r in enumerate(major):
        value = r["contribution"]
        y = pad + i * row
        bar = max(span * abs(value) / peak, 1.0)
        favours_home = value > 0
        color = HOME_INK if favours_home else AWAY_INK
        x = axis if favours_home else axis - bar
        label = r.get("_label") or FEATURE_LABELS.get(r["name"], r["name"])
        if r.get("imputed"):
            label += " (unknown)"

        # The two largest inputs are the comparison worth making, so they carry
        # the emphasis and everything below them recedes.
        lead = i < 2 and r["name"] != "_rest"
        opacity = "0.92" if lead else "0.45"

        if i % 2 == 0:
            parts.append(
                f'<rect x="0" y="{y:.1f}" width="{width}" height="{row}" '
                f'fill="var(--border)" opacity="0.16"/>'
            )
        parts.append(
            f'<text class="wfl{" lead" if lead else ""}" x="{label_w - 10}" '
            f'y="{y + row / 2 + 4:.1f}" text-anchor="end">{label}</text>'
        )
        parts.append(
            f'<rect x="{x:.1f}" y="{y + 5:.1f}" width="{bar:.1f}" '
            f'height="{row - 10}" rx="1.5" fill="{color}" opacity="{opacity}">'
            f'<title>{label}: {value:+.3f} log-odds toward '
            f'{home if favours_home else away}</title></rect>'
        )
        tx = axis + bar + 5 if favours_home else axis - bar - 5
        parts.append(
            f'<text class="wfv{" lead" if lead else ""}" x="{tx:.1f}" '
            f'y="{y + row / 2 + 4:.1f}" '
            f'text-anchor="{"start" if favours_home else "end"}">'
            f'{value:+.3f}</text>'
        )

    base = pad + row * len(major)
    parts.append(
        f'<line x1="{axis:.1f}" y1="{pad - 2}" x2="{axis:.1f}" y2="{base:.1f}" '
        f'stroke="var(--ink)" stroke-width="1" opacity="0.55"/>'
    )
    # Only name the directions the chart actually uses; a one-sided chart
    # labeled with both invites the reader to look for bars that are not there.
    if diverges:
        parts.append(
            f'<text class="sgl" x="{axis - 8:.1f}" y="{base + 15:.1f}" '
            f'text-anchor="end">&#9664; {away}</text>'
        )
        parts.append(
            f'<text class="sgl" x="{axis + 8:.1f}" y="{base + 15:.1f}">'
            f'{home} &#9654;</text>'
        )
    else:
        favoured = home if max(values) > 0 else away
        parts.append(
            f'<text class="sgl" x="{label_w:.1f}" y="{base + 15:.1f}">'
            f'every input favours {favoured} &#9654;</text>'
        )
    parts.append("</svg>")
    return Markup("".join(parts))


def margin_svg(
    projection: Any, home: str, away: str, *, width: int = 600, height: int = 210
) -> Markup:
    """The result as a margin, which is the readable form of the same simulation.

    This replaced a joint score grid. The grid was honest and nearly unreadable:
    a hundred cells, the largest near three percent, and no two distinguishable
    by eye. Collapsing to the margin keeps everything a reader actually asks --
    the win probability is the area on one side of the middle, one-run games are
    the two tallest central bars, blowouts are the tails -- and loses only the
    exact scoreline, which is the least reliable thing the model produces.
    """
    rows = projection.score.get("margin_distribution") or []
    if not rows:
        return Markup("")

    limit = max(abs(r["margin"]) for r in rows)
    slots = [m for m in range(-limit, limit + 1) if m != 0]
    lookup = {r["margin"]: r["p"] for r in rows}
    peak = max(lookup.values())

    left, right, top, bottom = 10, 10, 34, 34
    plot_w, plot_h = width - left - right, height - top - bottom
    step = plot_w / len(slots)
    mid = left + plot_w / 2

    home_p = projection.score["home_win_probability"]
    parts = [
        f'<svg class="mdist" '
        f'viewBox="0 0 {width} {height}" role="img" '
        f'aria-label="Winning margin distribution">'
    ]

    for i, m in enumerate(slots):
        p = lookup.get(m, 0.0)
        bar = plot_h * (p / peak)
        x = left + i * step
        y = top + plot_h - bar
        favours_home = m > 0
        color = HOME_INK if favours_home else AWAY_INK
        # One-run games are the single most likely outcome band in baseball and
        # the reason this model cannot do better; they get the emphasis.
        opacity = "0.95" if abs(m) == 1 else "0.6"
        edge = "&ge;" if m == limit else ("&le;" if m == -limit else "")
        parts.append(
            f'<rect x="{x + 1:.1f}" y="{y:.1f}" width="{step - 2:.1f}" '
            f'height="{bar:.1f}" rx="1.5" fill="{color}" opacity="{opacity}">'
            f'<title>{home if favours_home else away} by {edge}{abs(m)}: '
            f'{p * 100:.1f}%</title></rect>'
        )
        if abs(m) in (1, 3, 5, 7) or abs(m) == limit:
            parts.append(
                f'<text class="sgl" x="{x + step / 2:.1f}" '
                f'y="{top + plot_h + 13}" text-anchor="middle">'
                f'{edge}{abs(m)}</text>'
            )

    parts.append(
        f'<line x1="{mid:.1f}" y1="{top - 6}" x2="{mid:.1f}" '
        f'y2="{top + plot_h + 3:.1f}" stroke="var(--ink)" stroke-width="1.1" '
        f'opacity="0.5"/>'
    )
    # Win probability stated where its area is, rather than elsewhere on the page.
    parts += [
        f'<text class="mwin" x="{mid - 10:.1f}" y="{top - 18}" '
        f'text-anchor="end" fill="{AWAY_INK}">{away} {(1 - home_p) * 100:.0f}%</text>',
        f'<text class="mwin" x="{mid + 10:.1f}" y="{top - 18}" '
        f'fill="{HOME_INK}">{home} {home_p * 100:.0f}%</text>',
        f'<text class="sgl" x="{mid - 10:.1f}" y="{top - 6}" text-anchor="end">'
        f'&#9664; wins by</text>',
        f'<text class="sgl" x="{mid + 10:.1f}" y="{top - 6}">wins by &#9654;</text>',
    ]

    one_run = projection.score["p_one_run_game"]
    parts.append(
        f'<text class="sgl" x="{mid:.1f}" y="{height - 6}" text-anchor="middle">'
        f'{one_run * 100:.0f}% of the time it comes down to one run</text>'
    )
    parts.append("</svg>")
    return Markup("".join(parts))


def runs_by_side_svg(
    projection: Any, home: str, away: str, *, width: int = 520, height: int = 196
) -> Markup:
    """Each club's own run distribution, on one shared scale.

    Paired bars rather than a joint grid, for the same reason: the question a
    reader has is which offence is more likely to put up a number, and that is a
    comparison of two curves, not a hundred cells.
    """
    home_rows = {r["runs"]: r["p"] for r in projection.score.get("home_runs_distribution", [])}
    away_rows = {r["runs"]: r["p"] for r in projection.score.get("away_runs_distribution", [])}
    if not home_rows or not away_rows:
        return Markup("")

    limit = max(max(home_rows), max(away_rows))
    peak = max(list(home_rows.values()) + list(away_rows.values()))
    left, right, top, bottom = 10, 10, 22, 32
    plot_w, plot_h = width - left - right, height - top - bottom
    step = plot_w / (limit + 1)
    bar_w = (step - 3) / 2

    parts = [
        f'<svg class="rside" '
        f'viewBox="0 0 {width} {height}" role="img" '
        f'aria-label="Runs scored by each side">'
    ]
    for runs in range(limit + 1):
        x = left + runs * step
        for offset, rows, color, team in (
            (0, home_rows, HOME_INK, home), (bar_w + 1, away_rows, AWAY_INK, away)
        ):
            p = rows.get(runs, 0.0)
            bar = plot_h * (p / peak)
            parts.append(
                f'<rect x="{x + 1 + offset:.1f}" y="{top + plot_h - bar:.1f}" '
                f'width="{bar_w:.1f}" height="{bar:.1f}" rx="1" fill="{color}" '
                f'opacity="0.8"><title>{team} scores '
                f'{"9+" if runs == limit else runs}: {p * 100:.1f}%</title></rect>'
            )
        if runs % 2 == 0 or runs == limit:
            parts.append(
                f'<text class="sgl" x="{x + step / 2:.1f}" '
                f'y="{top + plot_h + 13}" text-anchor="middle">'
                f'{f"{runs}+" if runs == limit else runs}</text>'
            )

    for i, (team, color, mu) in enumerate((
        (home, HOME_INK, projection.score["exp_home_runs"]),
        (away, AWAY_INK, projection.score["exp_away_runs"]),
    )):
        x = left + i * 210
        parts.append(
            f'<rect x="{x}" y="{height - 13}" width="9" height="7" rx="1" '
            f'fill="{color}" opacity="0.8"/>'
        )
        parts.append(
            f'<text class="sgl" x="{x + 13}" y="{height - 6}">'
            f'{team} &middot; {mu:.2f} expected</text>'
        )
    parts.append(
        f'<text class="sgl" x="{width - right}" y="{top - 8}" text-anchor="end">'
        f'runs scored</text>'
    )
    parts.append("</svg>")
    return Markup("".join(parts))


def totals_svg(projection: Any, *, width: int = 520, height: int = 204) -> Markup:
    """Tonight's total-runs distribution against a normal night.

    Drawn over the measured league distribution because the projected total is
    close to unreadable alone: the model's expected total varies far less than
    real scoring does, so 8.4 looks much like 9.4 without the reference in the
    same frame. The gap between the two shapes is the part worth seeing.
    """
    rows = [r for r in projection.score["total_distribution"] if 2 <= r["total"] <= 20]
    if not rows:
        return Markup("")

    league = (projection.reference or {}).get("total_runs") or {}
    lo, hi = 2, 20
    peak = max(
        [r["p"] for r in rows]
        + [league.get(str(t), 0.0) for t in range(lo, hi + 1)]
    )
    left, right, top, bottom = 10, 10, 18, 32
    plot_w, plot_h = width - left - right, height - top - bottom
    step = plot_w / (hi - lo + 1)

    def x_of(total: float) -> float:
        return left + (total - lo) * step + step / 2

    parts = [
        f'<svg class="tdist" '
        f'viewBox="0 0 {width} {height}" role="img" '
        f'aria-label="Total runs tonight versus a normal game">'
    ]
    for r in rows:
        bar = plot_h * (r["p"] / peak)
        parts.append(
            f'<rect x="{x_of(r["total"]) - step / 2 + 1:.1f}" '
            f'y="{top + plot_h - bar:.1f}" width="{step - 2:.1f}" '
            f'height="{bar:.1f}" rx="1" fill="var(--accent)" opacity="0.7">'
            f'<title>{r["total"]} total runs: {r["p"] * 100:.1f}%</title></rect>'
        )
    if league:
        pts = " ".join(
            f"{x_of(t):.1f},{top + plot_h - plot_h * league.get(str(t), 0.0) / peak:.1f}"
            for t in range(lo, hi + 1)
        )
        parts.append(
            f'<polyline points="{pts}" fill="none" stroke="var(--ink)" '
            f'stroke-width="1.4" stroke-dasharray="3 2" opacity="0.6"/>'
        )

    expected = projection.score["expected_total"]
    league_mean = (projection.reference or {}).get("total_runs_mean")
    ex = x_of(expected)
    parts.append(
        f'<line x1="{ex:.1f}" y1="{top}" x2="{ex:.1f}" y2="{top + plot_h:.1f}" '
        f'stroke="{FLAG_INK}" stroke-width="1.6"/>'
    )
    parts.append(
        f'<text class="sga" x="{ex:.1f}" y="{top - 5}" text-anchor="middle" '
        f'fill="{FLAG_INK}">{expected:.1f}</text>'
    )
    for value in range(4, hi + 1, 4):
        parts.append(
            f'<text class="sgl" x="{x_of(value):.1f}" y="{top + plot_h + 13}" '
            f'text-anchor="middle">{value}</text>'
        )

    legend = height - 6
    parts.append(
        f'<rect x="{left}" y="{legend - 7}" width="9" height="7" rx="1" '
        f'fill="var(--accent)" opacity="0.7"/>'
    )
    parts.append(f'<text class="sgl" x="{left + 13}" y="{legend}">tonight</text>')
    parts.append(
        f'<line x1="{left + 62}" y1="{legend - 4}" x2="{left + 76}" '
        f'y2="{legend - 4}" stroke="var(--ink)" stroke-width="1.4" '
        f'stroke-dasharray="3 2" opacity="0.6"/>'
    )
    parts.append(
        f'<text class="sgl" x="{left + 80}" y="{legend}">a normal game'
        + (f" ({league_mean:.1f})" if league_mean else "")
        + "</text>"
    )
    parts.append("</svg>")
    return Markup("".join(parts))


def rate_compare(tonight: float, typical: float, *, width: int = 110) -> Markup:
    """One derived probability against its league rate, on a shared scale.

    A blowout chance of 26% means nothing without knowing 28% is normal. The
    direction of the gap is the finding, and it is invisible in the bare number.
    """
    scale = max(tonight, typical, 0.01) * 1.4
    a, b = width * tonight / scale, width * typical / scale
    return Markup(
        f'<svg class="rcmp" width="{width}" height="14" viewBox="0 0 {width} 14" '
        f'role="img" aria-label="tonight {tonight * 100:.0f}%, '
        f'normally {typical * 100:.0f}%">'
        f'<rect x="0" y="3" width="{a:.1f}" height="8" rx="1.5" '
        f'fill="var(--accent)" opacity="0.75"/>'
        f'<line x1="{b:.1f}" y1="0" x2="{b:.1f}" y2="14" stroke="var(--ink)" '
        f'stroke-width="1.4" opacity="0.65"/>'
        f"</svg>"
    )


def elo_scale_svg(projection: Any, home: str, away: str, *, width: int = 520) -> Markup:
    """Both clubs on the rating scale, with the league centered.

    Kept prominent because it earns it: the team rating supplies 59% of the
    movement in the win model, more than every starter input combined.
    """
    height = 52
    ratings = [projection.home.elo, projection.away.elo, 1500.0]
    lo, hi = min(ratings) - 45, max(ratings) + 45
    span = max(hi - lo, 1e-6)
    left, right = 14, width - 14
    axis = 32

    def x_of(rating: float) -> float:
        return left + (right - left) * (rating - lo) / span

    parts = [
        f'<svg class="eloscale" '
        f'viewBox="0 0 {width} {height}" role="img" aria-label="Team ratings">',
        f'<line x1="{left}" y1="{axis}" x2="{right}" y2="{axis}" '
        f'stroke="var(--border)" stroke-width="2" stroke-linecap="round"/>',
        f'<line x1="{x_of(1500):.1f}" y1="{axis - 7}" x2="{x_of(1500):.1f}" '
        f'y2="{axis + 7}" stroke="var(--muted)" stroke-width="1"/>',
        f'<text class="sgl" x="{x_of(1500):.1f}" y="{axis + 18}" '
        f'text-anchor="middle">league average</text>',
    ]
    for rating, label, color in (
        (projection.home.elo, home, HOME_INK),
        (projection.away.elo, away, AWAY_INK),
    ):
        x = x_of(rating)
        parts.append(f'<circle cx="{x:.1f}" cy="{axis}" r="5.5" fill="{color}"/>')
        parts.append(
            f'<text class="elolab" x="{x:.1f}" y="{axis - 13}" '
            f'text-anchor="middle" fill="{color}">{label} {rating:.0f}</text>'
        )
    parts.append("</svg>")
    return Markup("".join(parts))


def accuracy_bar(accuracy: float, baseline: float, *, width: int = 140) -> Markup:
    """One season's accuracy against the always-pick-home baseline."""
    scale = 0.70
    a = width * min(accuracy, scale) / scale
    b = width * min(baseline, scale) / scale
    return Markup(
        f'<svg class="accbar" width="{width}" height="14" viewBox="0 0 {width} 14" '
        f'role="img" aria-label="{accuracy * 100:.1f}% versus '
        f'{baseline * 100:.1f}% baseline">'
        f'<rect x="0" y="4" width="{a:.1f}" height="6" rx="1.5" '
        f'fill="var(--accent)" opacity="0.8"/>'
        f'<line x1="{b:.1f}" y1="1" x2="{b:.1f}" y2="13" stroke="var(--ink)" '
        f'stroke-width="1.4" opacity="0.65"/>'
        f"</svg>"
    )


# ---------------------------------------------------------------------------
# First five innings, player props, starter strikeouts
# ---------------------------------------------------------------------------


def three_way_bar(first_five: Any, home: str, away: str, *, width: int = 600) -> Markup:
    """Home leads, level, away leads — as one bar of three parts.

    A first-five result is genuinely three-way. 15.0% of games are tied after
    five, where a full game has none, so a two-sided bar would have to hide or
    reassign one outcome in six. Drawing the tie as its own band is the honest
    shape and happens to be the most interesting band on the chart.
    """
    if first_five is None:
        return Markup("")

    height, label = 34, 20
    parts = [
        f'<svg class="twbar" viewBox="0 0 {width} {height + label}" role="img" '
        f'aria-label="First five innings result">'
    ]
    segments = (
        (first_five.away_leads, AWAY_INK, f"{away} {first_five.away_leads * 100:.0f}%"),
        (first_five.tied, "var(--muted)", f"level {first_five.tied * 100:.0f}%"),
        (first_five.home_leads, HOME_INK, f"{home} {first_five.home_leads * 100:.0f}%"),
    )
    x = 0.0
    for share, color, text in segments:
        span = width * max(share, 0.0)
        parts.append(
            f'<rect x="{x:.1f}" y="0" width="{max(span, 1):.1f}" height="{height}" '
            f'fill="{color}" opacity="0.85"><title>{text}</title></rect>'
        )
        if span > 58:
            parts.append(
                f'<text class="twlab" x="{x + span / 2:.1f}" y="{height / 2 + 4:.0f}" '
                f'text-anchor="middle">{text}</text>'
            )
        x += span

    parts.append(
        f'<text class="sgl" x="0" y="{height + 14}">runs through five: '
        f'{first_five.expected_home:.2f} {home} &middot; '
        f'{first_five.expected_away:.2f} {away}</text>'
    )
    parts.append(
        f'<text class="sgl" x="{width}" y="{height + 14}" text-anchor="end">'
        f'{first_five.expected_total:.2f} total</text>'
    )
    parts.append("</svg>")
    return Markup("".join(parts))


def strikeout_distribution(
    prop: Any, *, width: int = 380, height: int = 150
) -> Markup:
    """The whole distribution of a starter's strikeout total, not just its mean.

    A projection of "5.8 strikeouts" is a number no pitcher can record. What a
    reader can act on is the shape: where the mass sits, and how much of it
    falls either side of the half-integer line a book would set.
    """
    if prop is None or not prop.distribution:
        return Markup("")

    counts = sorted(prop.distribution)
    lo, hi = min(counts), min(max(counts), 14)
    peak = max(prop.distribution.values())
    left, right, top, bottom = 10, 10, 22, 30
    plot_w, plot_h = width - left - right, height - top - bottom
    step = plot_w / max(hi - lo + 1, 1)

    parts = [
        f'<svg class="kdist" viewBox="0 0 {width} {height}" role="img" '
        f'aria-label="Strikeout total distribution">'
    ]
    line = prop.line
    for k in range(lo, hi + 1):
        p = prop.distribution.get(k, 0.0)
        bar = plot_h * (p / peak) if peak else 0
        x = left + (k - lo) * step
        # Colour by which side of the line the outcome falls on.
        over = k > line
        parts.append(
            f'<rect x="{x + 1:.1f}" y="{top + plot_h - bar:.1f}" '
            f'width="{step - 2:.1f}" height="{bar:.1f}" rx="1" '
            f'fill="{HOME_INK if over else "var(--muted)"}" '
            f'opacity="{0.85 if over else 0.5}">'
            f'<title>exactly {k}: {p * 100:.1f}%</title></rect>'
        )
        if k % 2 == 0 or k == lo:
            parts.append(
                f'<text class="sgl" x="{x + step / 2:.1f}" '
                f'y="{top + plot_h + 13}" text-anchor="middle">{k}</text>'
            )

    marker = left + (line - lo + 0.5) * step
    parts.append(
        f'<line x1="{marker:.1f}" y1="{top - 4}" x2="{marker:.1f}" '
        f'y2="{top + plot_h:.1f}" stroke="{FLAG_INK}" stroke-width="1.6"/>'
    )
    parts.append(
        f'<text class="sga" x="{marker:.1f}" y="{top - 8}" text-anchor="middle" '
        f'fill="{FLAG_INK}">{line:g}</text>'
    )
    over_line = prop.at_least(int(line) + 1)
    parts.append(
        f'<text class="sgl" x="{left}" y="{height - 6}">expected '
        f'{prop.expected:.2f} over {prop.batters_faced:.0f} batters</text>'
    )
    parts.append(
        f'<text class="sgl" x="{width - right}" y="{height - 6}" text-anchor="end">'
        f'over {line:g}: {over_line * 100:.0f}%</text>'
    )
    parts.append("</svg>")
    return Markup("".join(parts))


def prop_bar(value: float, *, width: int = 88, reference: float | None = None) -> Markup:
    """A probability as a bar, optionally against a reference rate."""
    value = max(0.0, min(1.0, float(value or 0.0)))
    parts = [
        f'<svg class="pbarmini" viewBox="0 0 {width} 12" role="img" '
        f'aria-label="{value * 100:.0f} percent">',
        f'<rect x="0" y="2" width="{width}" height="8" rx="1.5" '
        f'fill="var(--border)" opacity="0.5"/>',
        f'<rect x="0" y="2" width="{width * value:.1f}" height="8" rx="1.5" '
        f'fill="var(--accent)" opacity="0.8"/>',
    ]
    if reference is not None:
        x = width * max(0.0, min(1.0, reference))
        parts.append(
            f'<line x1="{x:.1f}" y1="0" x2="{x:.1f}" y2="12" '
            f'stroke="var(--ink)" stroke-width="1.2" opacity="0.6"/>'
        )
    parts.append("</svg>")
    return Markup("".join(parts))


def hit_spread(prop: Any, *, width: int = 108, height: int = 14) -> Markup:
    """How many hits, as a stack of the count probabilities.

    Reads left to right as none, one, two, three or more. The point is that a
    batter projected at 1.2 hits is not going to get 1.2 hits, and the widths
    show which outcomes are actually in play.
    """
    if not prop:
        return Markup("")
    distribution = prop.get("distribution") or {}
    if not distribution:
        return Markup("")

    shades = ["var(--border)", "#9db4cc", HOME_INK, FLAG_INK]
    parts = [
        f'<svg class="hspread" viewBox="0 0 {width} {height}" role="img" '
        f'aria-label="hit count probabilities">'
    ]
    x = 0.0
    for index in range(4):
        share = (
            distribution.get(index, 0.0) if index < 3
            else sum(v for k, v in distribution.items() if k >= 3)
        )
        span = width * share
        if span <= 0.4:
            continue
        parts.append(
            f'<rect x="{x:.1f}" y="1" width="{span:.1f}" height="{height - 2}" '
            f'fill="{shades[index]}" opacity="0.85">'
            f'<title>{"3+" if index == 3 else index} hits: {share * 100:.0f}%</title>'
            f'</rect>'
        )
        x += span
    parts.append("</svg>")
    return Markup("".join(parts))


# ---------------------------------------------------------------------------
# Link previews
# ---------------------------------------------------------------------------
# When the published URL is pasted into Slack, iMessage, Discord or anywhere
# else, the unfurler reads Open Graph tags out of the document head. Without
# them the link renders as a bare storage.googleapis.com path, which tells a
# reader nothing about which game it is.
#
# The description is built from the same figures the report is built from --
# records, the two starters, the model's call -- rather than a fixed sentence.
# A preview that says "Guardians scouting report" for every game is a label; one
# that says who is pitching and who is favoured is the reason to open it.

# Unfurlers truncate, and they do it mid-word. Slack shows roughly 300
# characters, iMessage far fewer, so the important half goes first.
PREVIEW_DESCRIPTION_LIMIT = 300


def preview_title(bundle) -> str:
    """The headline an unfurler shows.

    Full club names rather than abbreviations: "CLE @ COL" is fine as a browser
    tab, where the reader already knows what they opened, and useless in a chat
    window where they do not.
    """
    away = getattr(bundle.away, "name", "") or bundle.away.abbreviation
    home = getattr(bundle.home, "name", "") or bundle.home.abbreviation
    # `%-d` strips the leading zero on Linux and raises on Windows, which is
    # where this runs. The day number formats itself.
    day = bundle.game_date
    return f"{away} at {home} — {day:%b} {day.day}, {day.year}"


def preview_description(bundle) -> str:
    """One sentence of why this game is worth opening.

    Assembled in priority order and truncated at a word boundary, because the
    surfaces that show this cut it off without warning and a sentence that ends
    mid-number reads as broken rather than as trimmed.
    """
    parts: list[str] = []

    records = []
    for section in (bundle.away, bundle.home):
        profile = getattr(section, "profile", None)
        record = getattr(profile, "record", None) if profile else None
        if record is not None:
            records.append(f"{section.abbreviation} {record.wins}-{record.losses}")
    if len(records) == 2:
        parts.append(" vs ".join(records))

    starters = []
    for section in (bundle.away, bundle.home):
        found = next((p for p in getattr(section, "pitchers", []) or []
                      if getattr(p, "is_probable_starter", False)), None)
        if found is not None and getattr(found, "name", ""):
            starters.append(str(found.name).split(" (")[0])
    if len(starters) == 2:
        parts.append(f"{starters[0]} vs {starters[1]}")

    projection = getattr(bundle, "projection", None)
    if projection is not None:
        try:
            win = float(projection.win_probability)
            favourite = (bundle.home.abbreviation if win >= 0.5
                         else bundle.away.abbreviation)
            parts.append(f"model favours {favourite} at {max(win, 1 - win):.0%}")
        except (TypeError, ValueError, AttributeError):
            pass

    venue = getattr(bundle, "venue_name", "")
    if venue:
        parts.append(venue)

    text = " · ".join(parts)
    if len(text) <= PREVIEW_DESCRIPTION_LIMIT:
        return text
    clipped = text[:PREVIEW_DESCRIPTION_LIMIT].rsplit(" ", 1)[0]
    return clipped.rstrip(" ·") + "…"


def preview_url(bundle, *, bucket: str = "") -> str:
    """The canonical published address, or empty when nothing is configured.

    Built from the same rule `publish.object_name_for` uses. The two are kept in
    step by a test rather than by hope: a canonical URL pointing somewhere the
    file is not is worse than no canonical URL, because an unfurler will follow
    it and cache the 404.
    """
    if not bucket:
        return ""
    matchup = f"{bundle.away.abbreviation}-at-{bundle.home.abbreviation}"
    name = f"{bundle.game_date.isoformat()}_{matchup}.html"
    return f"https://storage.googleapis.com/{bucket}/reports/{name}"

def preview_image_url(bundle, *, bucket: str = "") -> str:
    """Where the drawn card will live once published.

    Same naming rule as the report and the same bucket, so the pair travel
    together. Empty without a bucket: an `og:image` pointing nowhere renders a
    broken thumbnail, which is worse than the text card it replaces.
    """
    if not bucket:
        return ""
    matchup = f"{bundle.away.abbreviation}-at-{bundle.home.abbreviation}"
    name = f"{bundle.game_date.isoformat()}_{matchup}.png"
    return f"https://storage.googleapis.com/{bucket}/reports/{name}"
