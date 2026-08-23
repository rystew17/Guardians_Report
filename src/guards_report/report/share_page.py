"""A page whose only job is to unfurl.

The report is 2.7 MB. The preview tags sit at byte 442, so any crawler that
reads part of the document finds them -- but a crawler that refuses the
document on Content-Length never opens it at all, and X is widely reported to
abandon large pages.

So the same tags are published a second time on a page nothing can object to:
about two kilobytes, no stylesheet, no script beyond a redirect. A person who
opens it lands on the report; a crawler that opens it gets the card and stops.

A separate URL rather than a redirect in front of the report, because a
redirect would make the full document the thing the crawler ultimately fetches,
which is the problem being avoided.

This is insurance, not a diagnosis. The likelier explanation for a link that
will not unfurl is X's own crawl cache, which holds a negative result for about
a week keyed on the exact URL -- and that is cured by a query string, not by
anything in this file.
"""

from __future__ import annotations

import json
from html import escape
from pathlib import Path
from typing import Any

TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<meta name="description" content="{description}">
<meta property="og:type" content="article">
<meta property="og:site_name" content="Guardians Scouting Report">
<meta property="og:title" content="{title}">
<meta property="og:description" content="{description}">
<meta property="og:url" content="{report_url}">
{image_tags}<meta name="twitter:card" content="{card}">
<meta name="twitter:title" content="{title}">
<meta name="twitter:description" content="{description}">
<meta name="theme-color" content="#0c2340">
<link rel="canonical" href="{report_url}">
<meta http-equiv="refresh" content="0; url={report_url}">
</head>
<body style="font:16px/1.6 -apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;background:#f4f6f8;color:#131922;margin:0;padding:3rem 1.5rem;text-align:center">
<p style="margin:0 0 1rem">Opening the scouting report&hellip;</p>
<p style="margin:0"><a href="{report_url}" style="color:#0c2340">{title}</a></p>
<script>location.replace({report_url_json});</script>
</body>
</html>
"""

IMAGE_TAGS = """<meta property="og:image" content="{image}">
<meta property="og:image:width" content="1200">
<meta property="og:image:height" content="630">
<meta property="og:image:alt" content="{title}">
<meta name="twitter:image" content="{image}">
"""


def build(
    bundle: Any, *, output_dir: Path, report_url: str, image_url: str = "",
    title: str = "", description: str = "",
) -> Path | None:
    """Write the share page beside the report.

    Returns None without a report URL: the page exists to carry an absolute
    canonical link, and it has nothing to point at otherwise.
    """
    if not report_url:
        return None

    safe_title = escape(title, quote=True)
    safe_description = escape(description, quote=True)
    image_tags = ""
    if image_url:
        image_tags = IMAGE_TAGS.format(image=image_url, title=safe_title)

    html = TEMPLATE.format(
        title=safe_title,
        description=safe_description,
        report_url=report_url,
        report_url_json=json.dumps(report_url),
        image_tags=image_tags,
        card="summary_large_image" if image_url else "summary",
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    matchup = f"{bundle.away.abbreviation}-at-{bundle.home.abbreviation}"
    path = output_dir / f"{bundle.game_date.isoformat()}_{matchup}-share.html"
    path.write_text(html, encoding="utf-8")
    return path
