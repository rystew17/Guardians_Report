"""Batched generation: fewer calls, same isolation guarantees.

One call per subject spent more on the skill file than on the data -- 58% of a
run's input was the instructions, re-sent once per player. Batching subjects
into a single call halves that, and the risk it introduces (a figure drifting
from one player's note into another's) is already covered by the verifier: a
number from a different digest is not in this subject's registered values.
"""

from __future__ import annotations

import asyncio
import json

import pytest

from guards_report.analysis import agent
from guards_report.analysis.digest import Digest


def _digest(subject_id: str, label: str, **figures) -> Digest:
    d = Digest(kind="batter", subject_id=subject_id, label=label)
    d.put("name", label)
    d.put_group("form", figures or {"season": {"OPS": ".750"}})
    return d


# ---------------------------------------------------------------------------
# Prompt and parsing
# ---------------------------------------------------------------------------


def test_batch_prompt_carries_every_subject_id():
    batch = [_digest("batter-1", "A"), _digest("batter-2", "B")]
    prompt = agent._batch_prompt(batch)

    assert "batter-1" in prompt and "batter-2" in prompt
    assert "single JSON object" in prompt
    # The isolation instruction is the whole reason a batch is safe to send.
    assert "never carry a figure from one subject into another" in prompt


def test_parses_a_clean_json_reply():
    batch = [_digest("batter-1", "A"), _digest("batter-2", "B")]
    reply = json.dumps({"batter-1": "Note for A.", "batter-2": "Note for B."})

    assert agent._parse_batch(reply, batch) == {
        "batter-1": "Note for A.",
        "batter-2": "Note for B.",
    }


def test_parses_through_a_markdown_fence():
    """A correct answer in a code fence should not cost a retry."""
    batch = [_digest("batter-1", "A")]
    reply = '```json\n{"batter-1": "Note for A."}\n```'

    assert agent._parse_batch(reply, batch) == {"batter-1": "Note for A."}


def test_ignores_subjects_that_were_not_asked_for():
    batch = [_digest("batter-1", "A")]
    reply = json.dumps({"batter-1": "Mine.", "batter-99": "Not in this batch."})

    assert agent._parse_batch(reply, batch) == {"batter-1": "Mine."}


def test_unparseable_reply_yields_nothing_rather_than_guessing():
    batch = [_digest("batter-1", "A")]
    assert agent._parse_batch("I couldn't do that.", batch) == {}


# ---------------------------------------------------------------------------
# Batching behaviour
# ---------------------------------------------------------------------------


def _run(digests, monkeypatch, replies, calls):
    async def fake_call(prompt, *, model):
        calls.append(prompt)
        return replies.pop(0), {"input_tokens": 1000, "output_tokens": 200}

    monkeypatch.setattr(agent, "_call", fake_call)
    return asyncio.run(agent.generate(digests, model="sonnet"))


def test_one_call_per_batch_not_per_subject(monkeypatch):
    digests = [_digest(f"batter-{i}", f"P{i}") for i in range(agent.BATCH_SIZE * 2)]
    replies = [
        json.dumps({d.subject_id: f"Note {d.subject_id}." for d in digests[:agent.BATCH_SIZE]}),
        json.dumps({d.subject_id: f"Note {d.subject_id}." for d in digests[agent.BATCH_SIZE:]}),
    ]
    calls: list[str] = []

    results = _run(digests, monkeypatch, replies, calls)

    assert len(results) == len(digests)
    assert len(calls) == 2, "16 subjects should be 2 calls, not 16"


def test_cached_subjects_are_never_batched(monkeypatch):
    """Cache hits cost nothing, so they must not be sent to the model."""
    digests = [_digest("batter-1", "A"), _digest("batter-2", "B")]
    cached = agent.Analysis(
        subject_id="batter-1", kind="batter", label="A",
        fingerprint=digests[0].fingerprint, text="Cached note.",
        verification=agent.verify("Cached note.", digests[0].values),
        generated_at=agent.datetime.now(agent.timezone.utc), model="sonnet",
    )
    cache = {f"batter-1:{digests[0].fingerprint}": cached}

    calls: list[str] = []

    async def fake_call(prompt, *, model):
        calls.append(prompt)
        return json.dumps({"batter-2": "Fresh note."}), {}

    monkeypatch.setattr(agent, "_call", fake_call)
    results = asyncio.run(agent.generate(digests, model="sonnet", cache=cache))

    assert len(results) == 2
    assert len(calls) == 1
    assert "batter-1" not in calls[0], "a cached subject was re-sent"


def test_missing_subject_falls_back_to_its_own_call(monkeypatch):
    """A partial reply costs one retry, not the whole batch."""
    digests = [_digest("batter-1", "A"), _digest("batter-2", "B")]
    calls: list[str] = []

    async def fake_call(prompt, *, model):
        calls.append(prompt)
        if len(calls) == 1:
            return json.dumps({"batter-1": "Note A."}), {}   # B omitted
        return "Note B.", {}

    monkeypatch.setattr(agent, "_call", fake_call)
    results = asyncio.run(agent.generate(digests, model="sonnet"))

    by_id = {r.subject_id: r.text for r in results}
    assert by_id == {"batter-1": "Note A.", "batter-2": "Note B."}
    assert len(calls) == 2


# ---------------------------------------------------------------------------
# Isolation and interruption
# ---------------------------------------------------------------------------


def test_a_figure_from_another_subject_is_flagged(monkeypatch):
    """The batch safety property: cross-contamination cannot pass silently."""
    a = _digest("batter-1", "A", season={"OPS": ".812"})
    b = _digest("batter-2", "B", season={"OPS": ".645"})

    # The model attributes A's OPS to B.
    reply = json.dumps({
        "batter-1": "A is running an .812 OPS.",
        "batter-2": "B is running an .812 OPS.",
    })

    async def fake_call(prompt, *, model):
        return reply, {}

    monkeypatch.setattr(agent, "_call", fake_call)
    results = {r.subject_id: r for r in asyncio.run(agent.generate([a, b], model="sonnet"))}

    assert results["batter-1"].verification.ok
    assert not results["batter-2"].verification.ok
    assert ".812" in results["batter-2"].verification.unverified


def test_usage_limit_keeps_the_notes_already_produced(monkeypatch):
    digests = [_digest(f"batter-{i}", f"P{i}") for i in range(agent.BATCH_SIZE * 2)]
    calls: list[str] = []

    async def fake_call(prompt, *, model):
        calls.append(prompt)
        if len(calls) == 1:
            return json.dumps(
                {d.subject_id: "Note." for d in digests[:agent.BATCH_SIZE]}
            ), {}
        raise agent.UsageLimitReached("usage limit reached")

    monkeypatch.setattr(agent, "_call", fake_call)

    with pytest.raises(agent.PartialResult) as caught:
        asyncio.run(agent.generate(digests, model="sonnet"))

    # The first batch survives rather than being thrown away with the run.
    assert len(caught.value.analyses) == agent.BATCH_SIZE
    assert "usage limit" in caught.value.reason


def test_limit_errors_are_told_apart_from_other_failures():
    assert agent._is_limit_error(RuntimeError("429 Too Many Requests"))
    assert agent._is_limit_error(RuntimeError("usage limit reached"))
    assert not agent._is_limit_error(RuntimeError("Not logged in"))
