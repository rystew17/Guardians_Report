"""Every `var(--x)` in the report must resolve to a declared custom property.

An undefined `var()` does not fall back -- it makes the whole declaration
invalid at computed-value time, so the property silently resets. A `font:`
shorthand loses its size, weight and line-height together; a `border` becomes
`none`; an SVG `stroke` disappears entirely.

Nothing about that failure is visible in the source, which is how three
undeclared names once survived a review: fourteen `font:` shorthands, every row
rule on the projections tables, and the axis line of a chart that consequently
rendered as two dots in empty space.

The template's stylesheet and the Python that emits inline SVG are checked
together, because the SVG borrows the same variables and fails the same way.
"""

from __future__ import annotations

import re
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src" / "guards_report" / "report"
TEMPLATE = SRC / "templates" / "report.html"
RENDERER = SRC / "render_html.py"

USE = re.compile(r"var\(\s*(--[a-z0-9-]+)")
# Declarations only count inside a rule block, never inside a var() call.
DECLARE = re.compile(r"(?:^|[{;\s])(--[a-z0-9-]+)\s*:")


def declared() -> set[str]:
    css = TEMPLATE.read_text(encoding="utf-8")
    return {
        name
        for name in DECLARE.findall(css)
        # A name only appearing as `var(--x)` is a use, not a declaration.
        if not re.search(rf"var\(\s*{re.escape(name)}\s*:", css)
    }


def used() -> dict[str, set[str]]:
    return {
        path.name: set(USE.findall(path.read_text(encoding="utf-8")))
        for path in (TEMPLATE, RENDERER)
    }


def test_the_check_sees_a_real_palette():
    # Guards against the regexes silently matching nothing.
    names = declared()
    assert len(names) > 15, f"expected a full palette, found {sorted(names)}"
    assert "--ink" in names and "--sans" in names


def test_every_referenced_variable_is_declared():
    known = declared()
    missing = {
        f"{where}: {name}"
        for where, names in used().items()
        for name in sorted(names - known)
    }
    assert not missing, (
        "undefined var() silently voids the whole declaration:\n  "
        + "\n  ".join(sorted(missing))
    )
