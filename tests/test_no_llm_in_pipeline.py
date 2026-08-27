"""The computation layers must not be able to reach an LLM.

The project's founding constraint is that every number is produced by code,
never by a model. Comments and good intentions do not enforce that; this does.

The written-analysis layer under `analysis/` is the one place a model is
allowed, and it is downstream of everything here: it receives finished figures
and returns prose. If an import ever crosses from a metrics or store module
into that layer, this test fails.
"""

from __future__ import annotations

import ast
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src" / "guards_report"

# Layers that must stay purely computational.
# `projections` is here for the same reason as the rest: a projection is a
# statistical estimate, and the moment a model could write one, the number
# stops being checkable.
PURE_LAYERS = ("metrics", "store", "sources", "ingest", "projections")

# Anything that can produce generated text.
FORBIDDEN_ROOTS = {
    "anthropic",
    "claude_agent_sdk",
    "openai",
    "google.generativeai",
    "langchain",
    "transformers",
    "llama_cpp",
    "ollama",
}
FORBIDDEN_INTERNAL = "guards_report.analysis"


def _imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            found.add(node.module)
    return found


def _pure_modules() -> list[Path]:
    return [
        path
        for layer in PURE_LAYERS
        for path in (SRC / layer).rglob("*.py")
    ]


def test_pure_layers_exist():
    # Guards against the whole suite silently passing on an empty file list.
    modules = _pure_modules()
    assert len(modules) > 10, f"expected the pipeline modules, found {len(modules)}"


def test_computation_layers_never_import_an_llm():
    offenders = []
    for path in _pure_modules():
        for name in _imports(path):
            root = name.split(".")[0]
            if root in FORBIDDEN_ROOTS or name.startswith(FORBIDDEN_INTERNAL):
                offenders.append(f"{path.relative_to(SRC)} imports {name}")

    assert not offenders, (
        "calculations must be hard-coded, never model-generated:\n  "
        + "\n  ".join(offenders)
    )


def test_analysis_layer_does_not_compute_from_raw_sources():
    """The prose layer may read finished figures, never fetch or recompute."""
    offenders = []
    for path in (SRC / "analysis").rglob("*.py"):
        for name in _imports(path):
            if name.startswith(("guards_report.sources", "guards_report.ingest")):
                offenders.append(f"{path.relative_to(SRC)} imports {name}")

    assert not offenders, (
        "the analysis layer must receive computed figures, not derive them:\n  "
        + "\n  ".join(offenders)
    )


# ---------------------------------------------------------------------------
# Every fetch is governed
# ---------------------------------------------------------------------------

def test_no_module_reaches_the_network_around_the_rate_limiter():
    """The rule the pitcher game logs broke, now enforced.

    `sources.http` owns the process-wide spacing and the retries. A module that
    calls urllib or requests directly gets neither -- and the failure is silent
    rather than loud: six such requests at once got this project stopped
    mid-refresh with nothing raised, because a socket timeout does not fire on
    a connection that is dribbling bytes. It looked exactly like a slow build,
    for fifteen minutes, until the container was reclaimed underneath it.

    Three modules are exempt, and each carries its own spacing and its own
    retries -- the rule is that a call is governed, not that it goes through
    one particular function:

      sources/http.py       owns the limiter the rest of this rule points at
      projections/pitches.py  Savant, not the stats API: a different host with
                            different tolerances, gated by `_savant_wait` and
                            retried inside `fetch_range`
      odds/client.py        a metered key of our own, and a failure there is
                            caught and costs the report only its prices

    Anything else reaching the network is ungoverned by construction.
    """
    import re
    from pathlib import Path

    root = Path(__file__).resolve().parents[1] / "src" / "guards_report"
    allowed = {"sources/http.py", "odds/client.py", "projections/pitches.py"}
    pattern = re.compile(r"\b(urlopen|requests\.(get|post))\s*\(")

    offenders = []
    for path in root.rglob("*.py"):
        rel = path.relative_to(root).as_posix()
        if rel in allowed:
            continue
        for number, line in enumerate(
                path.read_text(encoding="utf-8").splitlines(), start=1):
            if line.lstrip().startswith("#"):
                continue
            if pattern.search(line):
                offenders.append(f"{rel}:{number}: {line.strip()}")

    assert not offenders, (
        "these reach the network without the shared limiter or retries:\n  "
        + "\n  ".join(offenders))
