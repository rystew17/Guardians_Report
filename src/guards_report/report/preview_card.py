"""The image a shared link unfurls into.

X renders a link card image-first: the headline sits on top of the picture, and
with no image there is nothing to draw, so it falls back to showing the bare
URL. Slack, iMessage and Discord will render a text-only card happily; X will
not. That is the whole reason this module exists.

Drawn rather than screenshotted. A screenshot of the report would need a
headless browser and would be unreadable at 300 pixels wide anyway -- what a
preview has to carry is the four things a reader decides on: who is playing,
when, who is pitching, and what the model thinks. Those fit comfortably.

Kept deliberately plain. This is a thumbnail seen at thumbnail size in a feed,
so it is a wordmark, two clubs, and three lines of figures -- not a miniature of
the report.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

# The size every unfurler expects for a large card. X wants at least 300x157
# and crops toward 2:1; 1200x630 is the shape Slack, Discord and Facebook all
# read without letterboxing.
WIDTH, HEIGHT = 1200, 630

# The report's own palette, so the card and the page it opens are recognisably
# the same thing.
NAVY = (12, 35, 64)
CREAM = (244, 246, 248)
INK = (19, 25, 34)
MUTED = (108, 122, 137)
ACCENT = (227, 25, 55)
RULE = (221, 227, 233)

# Windows first, since that is where this runs, then the usual Linux paths so a
# container build does not silently fall back to the bitmap default -- which
# ignores size entirely and renders a 64-point headline at about 11 points.
FONT_CANDIDATES = {
    "bold": (
        "C:/Windows/Fonts/segoeuib.ttf",
        "C:/Windows/Fonts/arialbd.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
    ),
    "regular": (
        "C:/Windows/Fonts/segoeui.ttf",
        "C:/Windows/Fonts/arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
    ),
}


@dataclass
class CardLines:
    """What the card says, already decided."""

    away: str = ""
    home: str = ""
    date: str = ""
    records: str = ""
    starters: str = ""
    call: str = ""
    venue: str = ""


def _font(kind: str, size: int):
    from PIL import ImageFont

    for path in FONT_CANDIDATES[kind]:
        if Path(path).exists():
            try:
                return ImageFont.truetype(path, size)
            except OSError:
                continue
    # The bitmap default ignores `size`, so a card built on it is legible but
    # badly proportioned. Better than raising: the alternative is no image, and
    # no image is the bug this module was written to fix.
    return ImageFont.load_default()


def lines_for(bundle: Any) -> CardLines:
    """Pull the card's content off the bundle."""
    lines = CardLines(
        away=getattr(bundle.away, "name", "") or bundle.away.abbreviation,
        home=getattr(bundle.home, "name", "") or bundle.home.abbreviation,
        venue=getattr(bundle, "venue_name", "") or "",
    )
    day = bundle.game_date
    lines.date = f"{day:%A, %B} {day.day}, {day.year}"

    records = []
    for section in (bundle.away, bundle.home):
        profile = getattr(section, "profile", None)
        record = getattr(profile, "record", None) if profile else None
        if record is not None:
            records.append(
                f"{section.abbreviation} {record.wins}-{record.losses}")
    lines.records = "   ·   ".join(records)

    starters = []
    for section in (bundle.away, bundle.home):
        found = next((p for p in getattr(section, "pitchers", []) or []
                      if getattr(p, "is_probable_starter", False)), None)
        if found is not None and getattr(found, "name", ""):
            starters.append(str(found.name).split(" (")[0])
    if len(starters) == 2:
        lines.starters = f"{starters[0]}  vs  {starters[1]}"

    projection = getattr(bundle, "projection", None)
    if projection is not None:
        try:
            win = float(projection.win_probability)
            favourite = (bundle.home.abbreviation if win >= 0.5
                         else bundle.away.abbreviation)
            lines.call = f"Model favours {favourite} at {max(win, 1 - win):.0%}"
        except (TypeError, ValueError, AttributeError):
            pass
    return lines


def _fit(draw, text: str, font_for, start: int, floor: int, room: int):
    """The largest size at which `text` fits in `room` pixels.

    Club names run from "Reds" to "Diamondbacks", so a fixed size either wastes
    half the card or overflows it. Shrinking to fit is the only version that
    works for all thirty.
    """
    size = start
    while size > floor:
        font = font_for(size)
        if draw.textlength(text, font=font) <= room:
            return font
        size -= 2
    return font_for(floor)


def build(bundle: Any, *, output_dir: Path) -> Path | None:
    """Draw the card and write it beside the report. None if Pillow is absent.

    Returning None rather than raising is deliberate: the image is an
    enhancement to the link, and a report that fails to build because a drawing
    library is missing would be a much worse outcome than a link that unfurls
    without a picture.
    """
    try:
        from PIL import Image, ImageDraw
    except ImportError:
        return None

    lines = lines_for(bundle)
    image = Image.new("RGB", (WIDTH, HEIGHT), CREAM)
    draw = ImageDraw.Draw(image)

    # A navy band across the top, the report's own accent colour, so the card
    # and the page read as one object.
    draw.rectangle([0, 0, WIDTH, 132], fill=NAVY)
    draw.text((64, 40), "GUARDIANS SCOUTING REPORT",
              font=_font("bold", 30), fill=CREAM)
    draw.text((64, 82), lines.date, font=_font("regular", 24), fill=(150, 170, 195))

    room = WIDTH - 128
    y = 190
    away_font = _fit(draw, lines.away, lambda s: _font("bold", s), 62, 34, room)
    draw.text((64, y), lines.away, font=away_font, fill=INK)
    y += away_font.size + 6

    at_font = _font("regular", 26)
    draw.text((64, y), "at", font=at_font, fill=MUTED)
    y += at_font.size + 10

    home_font = _fit(draw, lines.home, lambda s: _font("bold", s), 62, 34, room)
    draw.text((64, y), lines.home, font=home_font, fill=INK)
    y += home_font.size + 34

    draw.line([(64, y), (WIDTH - 64, y)], fill=RULE, width=2)
    y += 26

    for text, colour, size in ((lines.records, MUTED, 27),
                               (lines.starters, INK, 30),
                               (lines.call, ACCENT, 29)):
        if not text:
            continue
        font = _fit(draw, text, lambda s: _font("regular", s), size, 18, room)
        draw.text((64, y), text, font=font, fill=colour)
        y += font.size + 12

    if lines.venue:
        draw.text((64, HEIGHT - 58), lines.venue,
                  font=_font("regular", 22), fill=MUTED)

    output_dir.mkdir(parents=True, exist_ok=True)
    matchup = f"{bundle.away.abbreviation}-at-{bundle.home.abbreviation}"
    path = output_dir / f"{bundle.game_date.isoformat()}_{matchup}.png"
    image.save(path, "PNG", optimize=True)
    return path
