"""Open Graph tags, so a published link unfurls into something legible.

A pasted storage.googleapis.com URL names neither the teams nor the date. These
tags are what turn it into a card, and the two failure modes are both quiet: a
description that says the same thing for every game, and a canonical URL that
points somewhere the file is not.

The second is the dangerous one. An unfurler follows the canonical, and if it
404s the preview is worse than it would have been with no tag at all -- and it
caches that result.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

import pytest

from guards_report.publish import gcs
from guards_report.report import render_html


@dataclass
class _Record:
    wins: int = 63
    losses: int = 66


@dataclass
class _Profile:
    record: Any = field(default_factory=_Record)


@dataclass
class _Pitcher:
    name: str = "Tanner Bibee"
    is_probable_starter: bool = True


@dataclass
class _Section:
    abbreviation: str
    name: str
    pitchers: list = field(default_factory=lambda: [_Pitcher()])
    profile: Any = field(default_factory=_Profile)


@dataclass
class _Projection:
    win_probability: float = 0.567


@dataclass
class _Bundle:
    away: Any = field(default_factory=lambda: _Section("CLE", "Cleveland Guardians"))
    home: Any = field(default_factory=lambda: _Section(
        "COL", "Colorado Rockies", [_Pitcher(name="Gabriel Hughes")]))
    game_date: date = date(2026, 8, 22)
    venue_name: str = "Coors Field"
    projection: Any = field(default_factory=_Projection)


def test_the_canonical_url_points_where_publish_actually_puts_the_file():
    """The one that must not drift.

    `preview_url` and `object_name_for` build the same path by different code,
    so nothing but this keeps them in step. An unfurler follows the canonical
    and caches what it finds, so a wrong one is worse than an absent one.
    """
    bundle = _Bundle()
    url = render_html.preview_url(bundle, bucket="a-bucket")

    matchup = f"{bundle.away.abbreviation}-at-{bundle.home.abbreviation}"
    on_disk = Path(f"{bundle.game_date.isoformat()}_{matchup}.html")
    assert url.endswith(gcs.object_name_for(on_disk))
    assert url == f"https://storage.googleapis.com/a-bucket/{gcs.object_name_for(on_disk)}"


def test_no_bucket_yields_no_canonical_rather_than_a_broken_one():
    """A relative or half-built URL would unfurl worse than nothing."""
    assert render_html.preview_url(_Bundle(), bucket="") == ""


def test_the_title_names_the_clubs_in_full():
    """"CLE @ COL" is fine on a browser tab and useless in a chat window."""
    title = render_html.preview_title(_Bundle())
    assert "Cleveland Guardians" in title and "Colorado Rockies" in title
    assert "2026" in title and "Aug" in title


def test_the_title_carries_no_platform_specific_format_code():
    """`%-d` strips a leading zero on Linux and raises on Windows."""
    assert "%" not in render_html.preview_title(_Bundle())


def test_the_description_carries_facts_rather_than_a_label():
    """A preview that reads the same for every game is a label, not a preview."""
    text = render_html.preview_description(_Bundle())
    assert "63-66" in text and "50-78" not in text or "63-66" in text
    assert "Bibee" in text and "Hughes" in text
    assert "57%" in text
    assert "Coors Field" in text


def test_the_description_survives_missing_pieces():
    """Opening day has no record; a late scratch has no starter."""
    bundle = _Bundle()
    bundle.projection = None
    bundle.away.pitchers = []
    text = render_html.preview_description(bundle)
    assert text and "None" not in text


def test_the_description_is_trimmed_at_a_word_boundary():
    """Unfurlers cut without warning, and a sentence ending mid-number reads
    as broken rather than as trimmed."""
    bundle = _Bundle()
    bundle.venue_name = "A" * 400
    text = render_html.preview_description(bundle)
    assert len(text) <= render_html.PREVIEW_DESCRIPTION_LIMIT + 1
    assert text.endswith("…")


def test_a_different_game_produces_a_different_preview():
    """The whole point: two games must not unfurl identically."""
    first = _Bundle()
    second = _Bundle(
        away=_Section("NYY", "New York Yankees"),
        home=_Section("BOS", "Boston Red Sox", [_Pitcher(name="Someone Else")]),
        game_date=date(2026, 9, 1))
    assert render_html.preview_title(first) != render_html.preview_title(second)
    assert render_html.preview_description(first) != render_html.preview_description(second)


# ---------------------------------------------------------------------------
# The card image
# ---------------------------------------------------------------------------

def test_the_image_url_matches_where_publish_puts_the_card():
    """`og:image` pointing at a 404 renders a broken thumbnail.

    That is strictly worse than the text card it replaces, so the URL in the
    tag and the object the publisher uploads are pinned together here.
    """
    from guards_report.publish import gcs

    bundle = _Bundle()
    url = render_html.preview_image_url(bundle, bucket="a-bucket")
    matchup = f"{bundle.away.abbreviation}-at-{bundle.home.abbreviation}"
    card = Path(f"{bundle.game_date.isoformat()}_{matchup}.png")
    assert url.endswith(gcs.object_name_for(card))


def test_the_card_and_the_report_differ_only_by_extension():
    """`publish` finds the card with `path.with_suffix('.png')`.

    If the two naming rules ever drift the upload silently skips the image and
    the tag points at nothing.
    """
    bundle = _Bundle()
    html_url = render_html.preview_url(bundle, bucket="b")
    png_url = render_html.preview_image_url(bundle, bucket="b")
    assert html_url[:-len(".html")] == png_url[:-len(".png")]


def test_no_bucket_yields_no_image_tag():
    assert render_html.preview_image_url(_Bundle(), bucket="") == ""


def test_the_card_draws_at_the_size_unfurlers_expect(tmp_path):
    """1200x630 is what Slack, X, Discord and Facebook read without cropping."""
    pytest.importorskip("PIL")
    from PIL import Image

    from guards_report.report import preview_card

    path = preview_card.build(_Bundle(), output_dir=tmp_path)
    assert path is not None and path.suffix == ".png"
    with Image.open(path) as image:
        assert image.size == (preview_card.WIDTH, preview_card.HEIGHT)
        assert image.size == (1200, 630)


def test_the_card_carries_the_game_rather_than_a_template(tmp_path):
    pytest.importorskip("PIL")
    from guards_report.report import preview_card

    lines = preview_card.lines_for(_Bundle())
    assert "Cleveland Guardians" in lines.away
    assert "Colorado Rockies" in lines.home
    assert "63-66" in lines.records
    assert "Bibee" in lines.starters and "Hughes" in lines.starters
    assert "57%" in lines.call


def test_a_long_club_name_shrinks_rather_than_overflowing(tmp_path):
    """Club names run from "Reds" to "Diamondbacks".

    A fixed size either wastes half the card or runs off the edge, and the
    overflow is invisible until someone shares a Diamondbacks game.
    """
    pytest.importorskip("PIL")
    from PIL import Image, ImageDraw

    from guards_report.report import preview_card

    image = Image.new("RGB", (preview_card.WIDTH, preview_card.HEIGHT))
    draw = ImageDraw.Draw(image)
    room = preview_card.WIDTH - 128
    for name in ("Reds", "Arizona Diamondbacks", "W" * 40):
        font = preview_card._fit(
            draw, name, lambda s: preview_card._font("bold", s), 62, 34, room)
        # The floor can still overflow for absurd input; what must not happen is
        # the full-size font being handed back for a name that does not fit.
        if font.size > 34:
            assert draw.textlength(name, font=font) <= room, name


def test_a_missing_drawing_library_costs_the_image_not_the_report(monkeypatch, tmp_path):
    """The report must build on a machine with no raster library."""
    import builtins

    from guards_report.report import preview_card

    real_import = builtins.__import__

    def no_pil(name, *args, **kwargs):
        if name == "PIL" or name.startswith("PIL."):
            raise ImportError("no PIL")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_pil)
    assert preview_card.build(_Bundle(), output_dir=tmp_path) is None
