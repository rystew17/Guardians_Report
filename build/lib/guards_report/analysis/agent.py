"""Generate written analysis via the Claude Agent SDK.

Calls authenticate as the Claude subscription (verified: `~/.claude` holds a
`claudeAiOauth` credential, no API key is set), so they draw on the plan's usage
window rather than pay-as-you-go API billing. That is why the Agent SDK is used
here instead of the Messages API, which would require a billed API key.

The dollar figure the SDK reports is an API-equivalent valuation, not a charge.
Tokens are the meter that applies, so tokens are what this module records.

The model is given no tools and no filesystem access. It receives a JSON digest
of figures that were already computed and returns prose. It cannot fetch, read,
or calculate anything -- which is what keeps the project's rule intact that
every number is sourced or computed in code, never generated.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from pathlib import Path
from datetime import datetime, timezone
from typing import Any

from guards_report.analysis.digest import Digest
from guards_report.analysis.verify import Verification, verify

# The writing guidance lives in a skill file rather than in this module, so the
# instructions the model follows are a document a person can read and edit
# without touching Python. It is loaded and passed on every call rather than
# left for the model to discover: this is the only thing the model does here,
# so it should never run without it.
SKILL_PATH = (
    Path(__file__).resolve().parents[3]
    / ".claude" / "skills" / "scouting-note" / "SKILL.md"
)

# Used only if the skill file cannot be read. Deliberately terse -- it keeps a
# run alive, and the constraint that actually protects the report is the
# numeric verification downstream, not this text.
FALLBACK_PROMPT = """You write short scouting notes for a Cleveland Guardians game preview from a
JSON block of already-computed figures.

Answer four things: what the player is good at, what he is bad at, which way he
is trending, and how he matches up in today's game. Do not restate the table
that sits below your note -- cite a figure only as evidence, no more than four
in total. Quote every number exactly as given; never round, approximate, or
combine two into a range. Respect small samples. Three to five sentences, prose
only."""


def _load_skill() -> str:
    """The scouting-note skill text, with its YAML frontmatter stripped."""
    try:
        text = SKILL_PATH.read_text(encoding="utf-8")
    except OSError:
        return FALLBACK_PROMPT

    # Frontmatter is metadata for skill discovery, not instruction.
    if text.startswith("---"):
        parts = text.split("---", 2)
        if len(parts) == 3:
            text = parts[2]
    return text.strip() or FALLBACK_PROMPT


SYSTEM_PROMPT = _load_skill()


TASK_PROMPTS = {
    "matchup": (
        "Write the game-level read for this matchup. What decides it, and what "
        "should someone watch for? Weigh each club's quality against its "
        "current form, and say where the two starters are mismatched against "
        "the lineups they face."
    ),
    "pitcher": (
        "Write the scouting note for this pitcher: what he is good at, what he "
        "is vulnerable to, which way he is trending, and how he sets up "
        "against the lineup he faces today."
    ),
    "reliever": (
        "Write the scouting note for this reliever: what he gets outs with, "
        "where he is vulnerable, which way he is trending, and the spot in "
        "today's game he is most likely to be used in. Say plainly if his "
        "recent workload makes him unlikely to be available."
    ),
    "batter": (
        "Write the scouting note for this hitter: what he does well, what "
        "pitchers get him out with, which way he is trending, and how he "
        "projects against today's starter specifically."
    ),
    "bench": (
        "Write the short note for this bench hitter: what he does well, what "
        "he struggles with, and the situation he is most likely to be used in "
        "today given the starter's handedness. Keep it to two or three "
        "sentences -- he may not play."
    ),
}



@dataclass
class Analysis:
    """One written note, with everything needed to audit it."""

    subject_id: str
    kind: str
    label: str
    fingerprint: str
    text: str
    verification: Verification
    generated_at: datetime
    model: str
    from_cache: bool = False
    cost_usd: float | None = None
    usage: dict[str, Any] = field(default_factory=dict)


class AgentUnavailable(RuntimeError):
    """The Agent SDK could not be reached or is not authenticated."""


class PartialResult(Exception):
    """Carries the notes produced before a run had to stop.

    Raised rather than returned so a caller cannot mistake a partial run for a
    complete one, while still getting every note that was paid for.
    """

    def __init__(self, analyses: list["Analysis"], *, reason: str) -> None:
        super().__init__(f"{reason} after {len(analyses)} notes")
        self.analyses = analyses
        self.reason = reason


class UsageLimitReached(RuntimeError):
    """The subscription's usage window is exhausted.

    Distinct from AgentUnavailable: some notes were very likely produced before
    this fired, and they must be kept. Retrying later resumes from the cache.
    """


# Batch size. The skill is ~730 tokens and is re-sent with every call, so one
# call per subject spends more on the instructions than on the data. Eight
# subjects per call cuts total input roughly in half while keeping each
# response small enough to stay coherent and easy to parse.
BATCH_SIZE = 8

_LIMIT_MARKERS = (
    "usage limit", "rate limit", "rate_limit", "quota",
    "too many requests", "429", "overloaded",
)


def _is_limit_error(exc: Exception) -> bool:
    text = str(exc).lower()
    return any(marker in text for marker in _LIMIT_MARKERS)


def _sdk():
    try:
        from claude_agent_sdk import (
            AssistantMessage,
            ClaudeAgentOptions,
            ResultMessage,
            TextBlock,
            query,
        )
    except ImportError as exc:  # pragma: no cover - import guard
        raise AgentUnavailable(
            "claude-agent-sdk is not installed; run: pip install claude-agent-sdk"
        ) from exc
    return AssistantMessage, ClaudeAgentOptions, ResultMessage, TextBlock, query


def _batch_prompt(batch: list[Digest]) -> str:
    """One prompt covering several subjects, answered as keyed JSON.

    Each subject carries its own id, and the reply is keyed by that id, so a
    note can never be silently attached to the wrong player. Anything that does
    slip across subjects is caught downstream: a figure belonging to another
    player is not in this subject's registered values, so verify() flags it.
    """
    blocks = []
    for digest in batch:
        blocks.append(
            "\n".join([
                f"--- subject_id: {digest.subject_id}",
                TASK_PROMPTS[digest.kind],
                f"Subject: {digest.label}",
                "Data:",
                digest.to_json(),
            ])
        )

    header = "\n\n".join([
        f"Write a scouting note for each of the {len(batch)} subjects below.",
        "Treat every subject independently. Use only the data given under that "
        "subject; never carry a figure from one subject into another's note.",
        "Reply with a single JSON object and nothing else -- no prose before or "
        "after, no markdown fence. Each key is a subject_id exactly as given, "
        "and each value is that subject's note as a plain string.",
    ])
    return header + "\n\n" + "\n\n".join(blocks)


def _parse_batch(text: str, batch: list[Digest]) -> dict[str, str]:
    """Pull the per-subject notes out of the model's reply.

    Tolerant of a markdown fence or stray prose around the object, because a
    reply that is correct apart from its wrapper should not cost a retry.
    """
    body = text.strip()
    if body.startswith("```"):
        body = body.split("```", 2)[1] if body.count("```") >= 2 else body
        body = body[4:] if body.lower().startswith("json") else body

    start, end = body.find("{"), body.rfind("}")
    if start == -1 or end == -1:
        return {}
    try:
        parsed = json.loads(body[start:end + 1])
    except json.JSONDecodeError:
        return {}
    if not isinstance(parsed, dict):
        return {}

    wanted = {d.subject_id for d in batch}
    return {
        k: v.strip() for k, v in parsed.items()
        if k in wanted and isinstance(v, str) and v.strip()
    }


async def _call(prompt: str, *, model: str) -> tuple[str, dict[str, Any]]:
    """One Agent SDK call with no tools, returning the text plus usage."""
    AssistantMessage, ClaudeAgentOptions, ResultMessage, TextBlock, query = _sdk()

    options = ClaudeAgentOptions(
        system_prompt=SYSTEM_PROMPT,
        model=model,
        # `tools=[]` removes the tool *definitions*; `allowed_tools=[]` only
        # restricted their use, which left the whole schema set in context and
        # -- worse -- left the Skill tool discoverable. Measured on a real run,
        # the agent spent an entire extra turn calling Skill to load
        # scouting-note, which this module already injects as the system
        # prompt: the same text delivered twice, one round trip to find out.
        tools=[],
        allowed_tools=[],
        # Do not load the project's .claude directory. The cwd is the repo, so
        # otherwise its settings, CLAUDE.md and skills are pulled into every
        # call -- context this task has no use for.
        setting_sources=[],
        # Turning JSON into three sentences is not a reasoning problem, and the
        # measured run spent 18,000 characters of thinking to produce 3,000
        # characters of notes. A small allowance keeps the judgment without
        # paying for deliberation the task does not need.
        max_thinking_tokens=1024,
        permission_mode="dontAsk",
    )

    parts: list[str] = []
    usage: dict[str, Any] = {}
    try:
        async for message in query(prompt=prompt, options=options):
            if isinstance(message, AssistantMessage):
                for block in message.content:
                    if isinstance(block, TextBlock):
                        parts.append(block.text)
            elif isinstance(message, ResultMessage):
                raw = getattr(message, "usage", None) or {}
                # Almost nothing arrives as plain `input_tokens` -- the harness
                # prompt is cached, so the real input volume shows up under the
                # cache fields. Reading only input/output reported roughly half
                # the true figure, which is worse than reporting none.
                usage = {
                    # Reported for reference only. With subscription auth this
                    # is an API-equivalent valuation, not a charge -- tokens
                    # are the meter that actually applies.
                    "cost_usd": getattr(message, "total_cost_usd", None),
                    "duration_ms": getattr(message, "duration_ms", None),
                    "input_tokens": _token_field(raw, "input_tokens"),
                    "output_tokens": _token_field(raw, "output_tokens"),
                    "cache_read_tokens": _token_field(raw, "cache_read_input_tokens"),
                    "cache_write_tokens": _token_field(
                        raw, "cache_creation_input_tokens"
                    ),
                }
    except Exception as exc:
        if _is_limit_error(exc):
            raise UsageLimitReached(str(exc)) from exc
        raise AgentUnavailable(f"Agent SDK call failed: {exc}") from exc

    return "".join(parts).strip(), usage


def _token_field(raw: Any, name: str) -> int | None:
    if isinstance(raw, dict):
        value = raw.get(name)
    else:
        value = getattr(raw, name, None)
    return int(value) if isinstance(value, (int, float)) else None


async def _generate_one(digest: Digest, *, model: str) -> tuple[str, dict[str, Any]]:
    """Fallback path: one subject, one call.

    Used when a batch reply cannot be parsed, so a malformed response costs one
    retry for the affected subjects rather than losing their notes.
    """
    prompt = "\n\n".join([
        TASK_PROMPTS[digest.kind],
        f"Subject: {digest.label}",
        f"Data:\n{digest.to_json()}",
    ])
    return await _call(prompt, model=model)


async def generate(
    digests: list[Digest],
    *,
    model: str = "sonnet",
    cache: dict[str, Analysis] | None = None,
    concurrency: int = 4,
    on_progress: Any = None,
) -> list[Analysis]:
    """Generate a note per digest, reusing cached notes where figures match.

    The cache is keyed on the digest fingerprint, so a note is regenerated only
    when the numbers behind it actually changed. Re-rendering a report costs
    nothing; re-running a day whose games have not moved costs nothing.
    """
    cache = cache or {}
    results: list[Analysis] = []
    pending: list[Digest] = []
    total = len(digests)
    done = 0

    # Cached notes cost nothing, so settle them first and batch only the rest.
    for digest in digests:
        key = f"{digest.subject_id}:{digest.fingerprint}"
        if key in cache:
            cached = cache[key]
            cached.from_cache = True
            # Re-check the stored prose against the current digest rather than
            # trusting the verdict saved with it. The figures are identical --
            # that is what the fingerprint guarantees -- but the verifier
            # itself improves, and a note flagged by an older, blinder version
            # should clear without costing a regeneration.
            cached.verification = verify(cached.text, digest.values)
            results.append(cached)
            done += 1
            if on_progress:
                on_progress(done, total, f"{digest.label} (cached)")
        else:
            pending.append(digest)

    def build(digest: Digest, text: str, usage: dict[str, Any]) -> Analysis:
        return Analysis(
            subject_id=digest.subject_id,
            kind=digest.kind,
            label=digest.label,
            fingerprint=digest.fingerprint,
            text=text,
            verification=verify(text, digest.values),
            generated_at=datetime.now(timezone.utc),
            model=model,
            cost_usd=usage.get("cost_usd"),
            usage=usage,
        )

    batches = [
        pending[i:i + BATCH_SIZE] for i in range(0, len(pending), BATCH_SIZE)
    ]

    for batch in batches:
        try:
            text, usage = await _call(_batch_prompt(batch), model=model)
        except UsageLimitReached:
            # Keep everything produced so far. The caller saves these to the
            # cache, so a later run resumes instead of starting over.
            raise PartialResult(results, reason="usage limit reached") from None

        notes = _parse_batch(text, batch)

        # Split the batch's usage across the subjects it produced, so per-note
        # figures stay meaningful without double-counting the call.
        share = max(len(notes), 1)
        per_note = {
            "cost_usd": (usage.get("cost_usd") or 0) / share or None,
            "batched": len(batch),
        }
        for field in (
            "input_tokens", "output_tokens",
            "cache_read_tokens", "cache_write_tokens",
        ):
            per_note[field] = (usage.get(field) or 0) // share or None

        missing = [d for d in batch if d.subject_id not in notes]
        for digest in batch:
            if digest.subject_id in notes:
                results.append(build(digest, notes[digest.subject_id], per_note))
                done += 1
                if on_progress:
                    on_progress(done, total, digest.label)

        # A reply we could not parse costs one retry for the affected subjects
        # rather than losing their notes entirely.
        for digest in missing:
            try:
                one_text, one_usage = await _generate_one(digest, model=model)
            except UsageLimitReached:
                raise PartialResult(results, reason="usage limit reached") from None
            results.append(build(digest, one_text, one_usage))
            done += 1
            if on_progress:
                on_progress(done, total, f"{digest.label} (retried)")

    return results


def generate_sync(digests: list[Digest], **kwargs: Any) -> list[Analysis]:
    return asyncio.run(generate(digests, **kwargs))
