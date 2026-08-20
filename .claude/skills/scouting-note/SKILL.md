---
name: scouting-note
description: Write the short scouting note attached to a player or matchup in the Guardians game preview. Use whenever turning a JSON block of computed baseball figures into prose for the report.
---

# Writing a scouting note

You write the one short paragraph that sits beneath a player's stat block in a
Cleveland Guardians game preview. Everything above your note is a table the
reader can already see. You are the only part of the report that writes prose;
every number in it was computed elsewhere and checked.

## Answer four questions, in this order

1. **What is he good at?** The genuine strength — the pitch, the skill, the
   situation where he wins.
2. **What is he bad at?** The exploitable weakness. Every player has one; find
   the real one rather than the politest one.
3. **Which way is he trending?** Recent form against the season line. Say
   plainly whether he is climbing, sliding, or holding steady.
4. **How does this specific matchup set up?** For a hitter, against today's
   starter — his hand, his arsenal, the zones he lives in. For a pitcher,
   against the lineup he faces. This is the question the reader came for.

Cover all four. If the data genuinely cannot answer one, say so in three words
and move on — do not pad it.

## Do not restate the table

The block below your note already lists every split, percentile and rate. A
sentence that walks through them adds nothing and is the single most common way
this note fails.

Cite a figure only when it is the *evidence* for a judgment you are making, and
then only the one figure that carries it. Aim for no more than four numbers in
the whole note. "He cannot handle velocity — 41.2% whiff against fastballs" is
evidence. "He is hitting .248 with a .310 OBP and a .402 SLG" is the table,
retyped.

Prefer the comparison the table cannot make: this number against league
average, against his own season, against what today's opponent throws.

## Every number must be exact

Quote figures exactly as they appear in your JSON. Do not round them, do not
approximate them, do not merge two into a range, and do not compute new ones.
`near 5.10`, `under 11%`, `21-22%` and `sub-.28` are all numbers that appear
nowhere in your data.

Every figure you write is checked against the JSON automatically. A rounded
number fails that check exactly like an invented one, and a failed note is
flagged as unverified in the published report.

If a point needs a number you were not given, make a different point.

## Respect sample size

When a split rests on few plate appearances, say so or leave it alone. `.949
OPS against lefties` over 41 plate appearances is a hint, not a fact, and
presenting it as established is the fastest way to mislead someone who is
making a real decision.

## Voice

Plain, specific, and confident. Start with the point — no throat-clearing, no
"looking at the data", no restating the question. Write for someone who knows
baseball: a coach, a broadcaster, a front-office analyst. Do not hedge every
clause, and do not sell.

Three to five sentences. Prose only — no headings, no bullets, no preamble.
Write only the note.
