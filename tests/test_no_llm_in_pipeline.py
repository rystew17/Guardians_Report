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
