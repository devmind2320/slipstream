"""Small deterministic text helpers shared by the scheduler and the detectors.

Kept in one file on purpose: three of these are used by four modules each, and
spreading them out would buy nothing.
"""

from __future__ import annotations

import re
import unicodedata

# Sentence / clause boundaries. The scheduler cuts here and never mid-word.
_SENT_END = re.compile(r"(?<=[.!?])[\"')\]]*\s")
_CLAUSE_END = re.compile(r"(?<=[,;:])\s")

_WORD = re.compile(r"[A-Za-z0-9_']+")
_NUMBER = re.compile(r"(?<![\w.])[+-]?\d[\d,]*(?:\.\d+)?(?![\w])")

STOPWORDS = frozenset(
    """
a an the and or but if while of to in on at by for with from as is are was were be been being
this that these those it its it's he she they them his her their our your my we you i us
do does did done have has had having will would shall should can could may might must
not no nor so than then there here when where which who whom whose what why how all any both
each few more most other some such only own same too very s t just don now about into over
under again further once during before after above below up down out off between against
""".split()
)

HEDGES = frozenset(
    """
maybe perhaps possibly probably likely apparently seemingly presumably reportedly allegedly
approximately roughly around about typically generally usually often sometimes might could
may believe think suspect assume guess unclear uncertain unsure estimate estimated
""".split()
)


def normalise(text: str) -> str:
    return unicodedata.normalize("NFKC", text)


def tokens(text: str) -> list[str]:
    return _WORD.findall(text.lower())


def content_tokens(text: str) -> list[str]:
    return [t for t in tokens(text) if t not in STOPWORDS and len(t) > 1]


def numbers(text: str) -> set[str]:
    """Numeric literals, comma-stripped and trailing-zero normalised.

    Numbers are the highest-signal thing a model fabricates, so they get their
    own comparison path in groundedness.
    """
    out: set[str] = set()
    for raw in _NUMBER.findall(text):
        cleaned = raw.replace(",", "").lstrip("+")
        try:
            val = float(cleaned)
        except ValueError:
            continue
        out.add(f"{val:g}")
    return out


def split_sentences(text: str) -> list[str]:
    parts = [p.strip() for p in _SENT_END.split(text)]
    return [p for p in parts if p]


def find_boundary(buffer: str, min_chars: int) -> int:
    """Index just past the last safe cut point at or after ``min_chars``.

    Prefers a sentence end, falls back to a clause end, then to a word break.
    Returns 0 when the buffer holds no safe cut yet — the caller keeps buffering.
    """
    if len(buffer) < min_chars:
        return 0
    for pattern in (_SENT_END, _CLAUSE_END):
        best = 0
        for m in pattern.finditer(buffer):
            if m.end() >= min_chars:
                best = m.end()
                break
        if best:
            return best
    # No punctuation available: cut on the last whitespace so we never split a word.
    idx = buffer.rfind(" ", min_chars)
    return idx + 1 if idx > 0 else 0


def approx_tokens(text: str) -> int:
    """Token count proxy. ~4 chars/token, floored at the word count."""
    words = len(_WORD.findall(text))
    return max(words, len(text) // 4)


def entity_density(text: str) -> float:
    """Capitalised-token + numeric density, a cheap named-entity proxy (§7.4)."""
    toks = _WORD.findall(text)
    if not toks:
        return 0.0
    caps = sum(1 for t in toks[1:] if t[:1].isupper())
    nums = len(_NUMBER.findall(text))
    return min(1.0, (caps + nums) / len(toks))


def hedging_ratio(text: str) -> float:
    toks = tokens(text)
    if not toks:
        return 0.0
    return sum(1 for t in toks if t in HEDGES) / len(toks)


def overlap_ratio(claim: str, context: str) -> float:
    """Fraction of the claim's content tokens present in the context."""
    c = content_tokens(claim)
    if not c:
        return 1.0
    ctx = set(content_tokens(context))
    return sum(1 for t in c if t in ctx) / len(c)
