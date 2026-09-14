"""The CLI, the app, and the BigQuery schemas.

The two entry points are thin, which is exactly why they are worth pinning: a
thin layer is where a flag gets wired to the wrong thing and nothing downstream
notices. This project has already shipped that bug -- the app passed
`--with-analysis` on every build, which is the flag that routes to a language
model, and it kept calling one for weeks after the CLI default stopped. Nothing
failed. The reports were fine. They were just generated.

So most of what follows is about the argument surface rather than about
behaviour: which flags exist, what the default does, and that the path from a
URL to a file on disk cannot be talked into leaving the output directory.
"""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import pytest

from guards_report import cli


# ---------------------------------------------------------------------------
# Date parsing
# ---------------------------------------------------------------------------

def test_the_relative_dates_a_person_actually_types_are_understood():
    assert cli._parse_date("today") == date.today()
    assert cli._parse_date("tomorrow") == date.today() + timedelta(days=1)
    assert cli._parse_date("yesterday") == date.today() - timedelta(days=1)


def test_an_iso_date_is_taken_literally():
    assert cli._parse_date("2026-08-22") == date(2026, 8, 22)


def test_an_unreadable_date_is_refused_rather_than_guessed():
    """Guessing would build a report for the wrong game and say nothing."""
    for text in ("08/22/2026", "next tuesday", "", "2026-13-45"):
        with pytest.raises(ValueError):
            cli._parse_date(text)


# ---------------------------------------------------------------------------
# The analysis routing — where the model bug lived
# ---------------------------------------------------------------------------

def _build_parser_flags() -> set[str]:
    """Every flag the `build` subcommand accepts."""
    import argparse
    import io
    import contextlib

    flags: set[str] = set()
    real_add_parser = argparse._SubParsersAction.add_parser
    captured: list = []

    def spy(self, name, **kwargs):
        parser = real_add_parser(self, name, **kwargs)
        if name == "build":
            captured.append(parser)
        return parser

    argparse._SubParsersAction.add_parser = spy
    try:
        with contextlib.redirect_stdout(io.StringIO()), \
                contextlib.redirect_stderr(io.StringIO()):
            try:
                cli.main(["--help"])
            except SystemExit:
                pass
    finally:
        argparse._SubParsersAction.add_parser = real_add_parser

    for parser in captured:
        for action in parser._actions:
            flags.update(action.option_strings)
    return flags


def test_the_build_command_offers_both_analysis_switches():
    """`--with-analysis` selects the model; `--no-analysis` turns prose off.

    Neither is the default, which is the point: the default computes.
    """
    flags = _build_parser_flags()
    if not flags:
        pytest.skip("could not introspect the build parser")
    assert "--with-analysis" in flags
    assert "--no-analysis" in flags


def _string_literals(path: Path, function: str) -> set[str]:
    """Every string constant inside one function, docstrings excluded.

    Read from the syntax tree rather than the text. A substring search over the
    source matches the comment explaining the bug as readily as the bug, which
    is how the first version of this test failed on a correct file.
    """
    import ast

    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        # `async def` is a different node type, and the build command is one.
        if (isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                and node.name == function):
            body = node.body
            if (body and isinstance(body[0], ast.Expr)
                    and isinstance(body[0].value, ast.Constant)
                    and isinstance(body[0].value.value, str)):
                body = body[1:]          # drop the docstring
            found = set()
            for statement in body:
                for inner in ast.walk(statement):
                    if isinstance(inner, ast.Constant) and isinstance(inner.value, str):
                        found.add(inner.value)
            return found
    return set()


def test_the_app_does_not_pass_the_flag_that_routes_to_a_model():
    """The bug this file exists for.

    The app shelled out with `--with-analysis --model sonnet` on every build, so
    it kept calling a language model long after the CLI stopped. Nothing failed
    and no report looked wrong.
    """
    literals = _string_literals(
        Path(cli.__file__).parent / "app" / "main.py", "_run_build")
    assert literals, "could not read the build command"
    assert "--with-analysis" not in literals
    assert "--model" not in literals
    assert "--no-analysis" in literals, "the checkbox must turn prose off"


def test_the_computed_analysis_is_what_the_cli_reaches_for_by_default():
    """`--with-analysis` selects the model; falling through computes.

    Checked structurally: the model call must sit under the flag, and the
    computed call must sit under the branch that does not require it.
    """
    import ast

    tree = ast.parse(Path(cli.__file__).read_text(encoding="utf-8"))
    calls_under_flag: set[str] = set()
    calls_under_default: set[str] = set()

    for node in ast.walk(tree):
        if not isinstance(node, ast.If):
            continue
        test = ast.dump(node.test)
        if "with_analysis" not in test:
            continue
        for inner in ast.walk(node):
            if isinstance(inner, ast.Call) and isinstance(inner.func, ast.Name):
                calls_under_flag.add(inner.func.id)
        for clause in node.orelse:
            for inner in ast.walk(clause):
                if isinstance(inner, ast.Call) and isinstance(inner.func, ast.Name):
                    calls_under_default.add(inner.func.id)

    assert "attach_analysis" in calls_under_flag, "the model path lost its flag"
    assert "attach_computed_analysis" in calls_under_default, (
        "the default path must compute rather than generate")


# ---------------------------------------------------------------------------
# Serving a report by name
# ---------------------------------------------------------------------------

def test_a_report_name_cannot_climb_out_of_the_output_directory(tmp_path, monkeypatch):
    """The name arrives from a URL, so it is untrusted.

    Resolved and checked rather than joined and trusted, because `..` in a path
    segment is the oldest way to read a file the server never meant to serve.
    """
    from fastapi import HTTPException

    from guards_report.app import main

    secret = tmp_path / "secret.txt"
    secret.write_text("not a report", encoding="utf-8")
    out = tmp_path / "out"
    out.mkdir()
    (out / "real.html").write_text("<html></html>", encoding="utf-8")

    class _Settings:
        output_dir = out

    monkeypatch.setattr(main, "_settings", lambda: _Settings())

    assert main._safe_report("real.html").name == "real.html"
    for attempt in ("../secret.txt", "..\\secret.txt",
                    "subdir/../../secret.txt"):
        with pytest.raises(HTTPException) as raised:
            main._safe_report(attempt)
        assert raised.value.status_code == 404


def test_a_name_that_is_not_a_file_is_refused(tmp_path, monkeypatch):
    from fastapi import HTTPException

    from guards_report.app import main

    out = tmp_path / "out"
    out.mkdir()
    (out / "adir").mkdir()

    class _Settings:
        output_dir = out

    monkeypatch.setattr(main, "_settings", lambda: _Settings())
    for attempt in ("missing.html", "adir"):
        with pytest.raises(HTTPException):
            main._safe_report(attempt)


# ---------------------------------------------------------------------------
# BigQuery schemas
# ---------------------------------------------------------------------------

SCHEMA_NAMES = (
    "RAW_API_CALL", "DIM_PLAYER", "DIM_TEAM", "DIM_VENUE",
    "DIM_LEAGUE_CONSTANT", "FACT_GAME_SCHEDULE", "FACT_PLAYER_GAME_LOG",
    "FACT_PLAYER_SPLIT", "FACT_SAVANT_LEADERBOARD", "REPORT_RUN",
)


@pytest.mark.parametrize("name", SCHEMA_NAMES)
def test_every_schema_is_a_non_empty_list_of_fields(name):
    from guards_report.store import schemas

    schema = getattr(schemas, name)
    assert isinstance(schema, list) and schema, name


@pytest.mark.parametrize("name", SCHEMA_NAMES)
def test_no_schema_declares_the_same_column_twice(name):
    """BigQuery accepts a duplicate at definition time and fails at load, which
    puts the error a long way from its cause."""
    from guards_report.store import schemas

    fields = [field.name for field in getattr(schemas, name)]
    assert len(fields) == len(set(fields)), name


@pytest.mark.parametrize("name", SCHEMA_NAMES)
def test_every_field_declares_a_type_and_a_mode(name):
    from guards_report.store import schemas

    for field in getattr(schemas, name):
        assert field.field_type, f"{name}.{field.name}"
        assert field.mode in ("REQUIRED", "NULLABLE", "REPEATED"), (
            f"{name}.{field.name} has mode {field.mode!r}")


def test_a_required_field_is_one_the_source_always_sends():
    """A REQUIRED column that the source sometimes omits fails the whole load
    rather than the row, so the list is short by design."""
    from guards_report.store import schemas

    required = [f.name for f in schemas.RAW_API_CALL if f.mode == "REQUIRED"]
    assert "fetched_at" in required and "url" in required


# ---------------------------------------------------------------------------
# What a failed build says it was
# ---------------------------------------------------------------------------

def test_a_killed_build_says_it_ran_out_of_memory():
    """The failure that cost two evenings to name.

    On Cloud Run the container's memory limit counts the kernel page cache for
    files read through the GCS mount, so a build climbs for its whole run
    rather than settling, and gets shot partway through. The subprocess dies on
    signal 9 while the app survives to report it -- so the page showed "build
    exited with code -9" above a log that simply stopped, which explains
    nothing to somebody holding a phone. The exit code knew what happened; it
    only had to say so.
    """
    from guards_report.app import main

    said = main._why_it_died(-9, None)
    assert "memory" in said.lower(), said


def test_a_missing_game_is_not_reported_as_a_crash():
    """Exit 2 is the build declining a date it has no game for, which is a
    different thing from the build breaking."""
    from guards_report.app import main

    assert "no game" in main._why_it_died(2, None).lower()


def test_a_clean_exit_with_nothing_written_is_still_a_failure():
    from guards_report.app import main

    assert "without writing" in main._why_it_died(0, None)


def test_an_unrecognised_code_still_reports_the_number():
    """Guessing beyond the codes we actually know would be worse than the bare
    number it replaced."""
    from guards_report.app import main

    assert "7" in main._why_it_died(7, None)



# ---------------------------------------------------------------------------
# Watching a job, and finding the scripts it runs
# ---------------------------------------------------------------------------

def test_two_listeners_both_see_the_whole_run():
    """The bug that cost a build on a real card.

    One queue per job is single-consumer: two browsers -- or one that
    reconnected while the first stream was still draining -- competed for every
    line, so each reached exactly one of them. The terminator is a line like any
    other, so when it landed on the abandoned stream the live one never learned
    the build had finished and sat on keepalives until Cloud Run cut it an hour
    later. The logs showed both halves: one stream ending at 209s with the
    result, another running the full 3600s without it.
    """
    import asyncio
    from guards_report.app.main import Job

    job = Job(id="x", game_date="2026-09-14")
    first, second = asyncio.Queue(), asyncio.Queue()
    job.subscribers += [first, second]

    job.publish("building")
    job.publish("__DONE__ report.html")

    for queue in (first, second):
        assert queue.get_nowait() == "building"
        assert queue.get_nowait() == "__DONE__ report.html"


def test_published_lines_are_replayable_for_a_late_listener():
    import asyncio
    from guards_report.app.main import Job

    job = Job(id="x", game_date="2026-09-14")
    job.publish("one")
    late = asyncio.Queue()
    job.subscribers.append(late)
    job.publish("two")

    assert job.lines == ["one", "two"]      # the backlog a reconnect replays
    assert late.get_nowait() == "two"       # and only what it missed live


def test_the_refit_script_is_found_outside_a_source_tree():
    """`_root()` is `parents[3]`, which is the project root in a source tree and
    `site-packages`' parent once the package is installed. The refit pointed
    there and died on a missing file in every container run."""
    from pathlib import Path
    from guards_report.app import main

    found = main._script("refit.py")
    assert found is not None and found.is_file(), found
    assert found.name == "refit.py"
    assert main._script("no-such-script.py") is None


def test_a_missing_refit_script_is_a_failure_not_a_rejection():
    """Python exits 2 when it cannot open a file, and 2 is refit.py's own code
    for "measured and turned down". A missing file therefore reported itself as
    the guard working -- the most misleading answer available."""
    import asyncio
    from guards_report.app import main
    from guards_report.app.main import Job

    job = Job(id="r", game_date="refit")
    original = main._script
    main._script = lambda name: None
    try:
        asyncio.run(main._run_refit(job, force_outcome=False))
    finally:
        main._script = original

    assert job.status == "failed"
    assert "not found" in (job.error or "")
    assert any(l.startswith("__FAILED__") for l in job.lines), job.lines
