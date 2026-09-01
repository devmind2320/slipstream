"""PII detection. cost_class = floor.

INV-1 lives in this file. This detector runs on the prompt and on *every* output
chunk in *every* band, including LOW, and it is never skipped by a latency
budget or by degraded mode. If the risk estimator is wrong, this is what stops
the leak.

Default engine is regex + gazetteer + checksum validation: offline, deterministic,
sub-millisecond. Presidio + spaCy NER is layered on top when
SLIPSTREAM_USE_PRESIDIO=1 is set and the packages are importable; its findings are
merged, never substituted, so turning presidio off can only reduce recall of
free-text names, never of structured identifiers.

Raw values never leave this module. Callers get entity type, offsets and a salted
hash (§6).
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

from .. import settings
from ..contracts import EntityHit, GuardContext, RiskCategory, Verdict
from .base import DetectorBase


def salted_hash(value: str) -> str:
    return hashlib.sha256(
        (settings.ENTITY_SALT + "|" + value.strip().lower()).encode("utf-8")
    ).hexdigest()[:16]


@dataclass(frozen=True)
class Pattern:
    entity_type: str
    regex: re.Pattern[str]
    confidence: float = 0.9
    validator: str | None = None
    group: int = 0


def _luhn(digits: str) -> bool:
    ds = [int(c) for c in digits if c.isdigit()]
    if len(ds) < 12:
        return False
    total, parity = 0, len(ds) % 2
    for i, d in enumerate(ds):
        if i % 2 == parity:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return total % 10 == 0


# Verhoeff, used by UIDAI. A 12-digit string that passes this is very unlikely
# to be a coincidence; one that fails is still reported, at lower confidence,
# because a *fabricated* Aadhaar-shaped number is itself a privacy event.
_VERHOEFF_D = [
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [1, 2, 3, 4, 0, 6, 7, 8, 9, 5],
    [2, 3, 4, 0, 1, 7, 8, 9, 5, 6], [3, 4, 0, 1, 2, 8, 9, 5, 6, 7],
    [4, 0, 1, 2, 3, 9, 5, 6, 7, 8], [5, 9, 8, 7, 6, 0, 4, 3, 2, 1],
    [6, 5, 9, 8, 7, 1, 0, 4, 3, 2], [7, 6, 5, 9, 8, 2, 1, 0, 4, 3],
    [8, 7, 6, 5, 9, 3, 2, 1, 0, 4], [9, 8, 7, 6, 5, 4, 3, 2, 1, 0],
]
_VERHOEFF_P = [
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [1, 5, 7, 6, 2, 8, 3, 0, 9, 4],
    [5, 8, 0, 3, 7, 9, 6, 1, 4, 2], [8, 9, 1, 6, 0, 4, 3, 5, 2, 7],
    [9, 4, 5, 3, 1, 2, 6, 8, 7, 0], [4, 2, 8, 6, 5, 7, 3, 9, 0, 1],
    [2, 7, 9, 3, 8, 0, 6, 4, 1, 5], [7, 0, 4, 6, 9, 1, 3, 2, 5, 8],
]


def _verhoeff(digits: str) -> bool:
    ds = [int(c) for c in reversed(digits) if c.isdigit()]
    if len(ds) != 12:
        return False
    c = 0
    for i, d in enumerate(ds):
        c = _VERHOEFF_D[c][_VERHOEFF_P[i % 8][d]]
    return c == 0


PATTERNS: tuple[Pattern, ...] = (
    Pattern("EMAIL", re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]{2,}\b"), 0.98),
    # India: PAN is 5 letters, 4 digits, 1 letter.
    Pattern("PAN", re.compile(r"\b[A-Z]{5}[0-9]{4}[A-Z]\b"), 0.96),
    Pattern(
        "AADHAAR",
        re.compile(r"\b[2-9]\d{3}[ -]?\d{4}[ -]?\d{4}\b"),
        0.72,
        validator="verhoeff",
    ),
    Pattern("PHONE_IN", re.compile(r"(?:\+?91[- ]?)?\b[6-9]\d{9}\b"), 0.85),
    Pattern(
        "PHONE",
        re.compile(r"\+\d{1,3}[- ]?\(?\d{2,4}\)?[- ]?\d{3,4}[- ]?\d{3,4}\b"),
        0.8,
    ),
    Pattern("IBAN", re.compile(r"\b[A-Z]{2}\d{2}[A-Z0-9]{11,30}\b"), 0.9),
    Pattern("NATIONAL_ID", re.compile(r"\b\d{3}-\d{2}-\d{4}\b"), 0.93),
    Pattern(
        "CREDIT_CARD",
        re.compile(r"\b(?:\d[ -]?){12,18}\d\b"),
        0.95,
        validator="luhn",
    ),
    Pattern(
        "ACCOUNT",
        re.compile(
            r"\b(?:a/c|acct|account)\s*(?:no\.?|number|#)?\s*[:#]?\s*(\d{9,18})\b",
            re.I,
        ),
        0.94,
        group=1,
    ),
    Pattern(
        "COMPENSATION",
        re.compile(
            r"(?:(?:₹|INR|Rs\.?|\$|USD|€|EUR|£|GBP)\s?[\d,]+(?:\.\d{1,2})?"
            r"(?:\s?(?:lakh|lakhs|crore|crores|k|K|m|M|LPA|lpa))?)"
            r"|(?:\b[\d,]+(?:\.\d{1,2})?\s?(?:LPA|lpa|lakh|lakhs|crore|crores)\b)",
        ),
        0.66,
    ),
    Pattern(
        "DOB",
        re.compile(
            r"\b(?:\d{1,2}[-/]\d{1,2}[-/]\d{2,4}"
            r"|\d{1,2}\s(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s\d{4})\b"
        ),
        0.55,
    ),
    Pattern(
        "PERSON",
        re.compile(r"\b(?:Mr|Mrs|Ms|Dr|Prof)\.?\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?\b"),
        0.9,
    ),
    Pattern(
        "PERSON",
        re.compile(
            r"\b(?:employee|candidate|customer|client|patient|name)\s*(?:name)?\s*[:\-]\s*"
            r"([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2})",
            re.I,
        ),
        0.88,
        group=1,
    ),
    Pattern(
        "ADDRESS",
        re.compile(
            r"\b\d{1,4}[/-]?[A-Z]?,?\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\s+"
            r"(?:Road|Rd|Street|St|Lane|Ln|Avenue|Ave|Nagar|Marg|Colony|Sector)\b"
        ),
        0.7,
    ),
    Pattern("PIN_IN", re.compile(r"\b(?:PIN|pin ?code)\s*[:\-]?\s*(\d{6})\b", re.I), 0.75, group=1),
)

# A small gazetteer so free-text names are caught without spaCy. Deliberately
# small and documented as a limitation rather than dressed up as NER.
FIRST_NAMES = frozenset(
    """
priya rahul anita vikram sunil deepa arjun kavya rohit meera sanjay neha amit pooja
karthik divya suresh lakshmi manish shreya ajay ritu nikhil anjali harish swati
john sarah michael emma david olivia james sophia robert emily william grace
maria carlos ahmed fatima wei ling yuki hans anna
""".split()
)
_CAP_BIGRAM = re.compile(r"\b([A-Z][a-z]{2,})\s+([A-Z][a-z]{2,})\b")

# Precomputed for offset-preserving masking.
_MASK_FMT = "[{}]"


def _validate(p: Pattern, value: str) -> float:
    if p.validator == "luhn":
        return p.confidence if _luhn(value) else 0.0
    if p.validator == "verhoeff":
        # Shape match alone is enough to withhold: an Aadhaar-shaped number in an
        # outbound response is a privacy event whether or not it checksums.
        return p.confidence if _verhoeff(value.replace(" ", "").replace("-", "")) else 0.6
    return p.confidence


def scan(text: str) -> list[EntityHit]:
    """All entity hits in ``text``, de-overlapped, highest confidence wins."""
    raw: list[EntityHit] = []
    for p in PATTERNS:
        for m in p.regex.finditer(text):
            try:
                value = m.group(p.group)
                start, end = m.span(p.group)
            except (IndexError, re.error):  # pragma: no cover
                continue
            if not value:
                continue
            conf = _validate(p, value)
            if conf <= 0.0:
                continue
            raw.append(
                EntityHit(
                    entity_type=p.entity_type,
                    start=start,
                    end=end,
                    value_hash=salted_hash(value),
                    confidence=conf,
                )
            )

    for m in _CAP_BIGRAM.finditer(text):
        if m.group(1).lower() in FIRST_NAMES:
            raw.append(
                EntityHit(
                    entity_type="PERSON",
                    start=m.start(),
                    end=m.end(),
                    value_hash=salted_hash(m.group(0)),
                    confidence=0.82,
                )
            )

    if settings.USE_PRESIDIO:
        raw.extend(_presidio_hits(text))

    return _dedupe(raw)


def _dedupe(hits: list[EntityHit]) -> list[EntityHit]:
    """Drop spans contained in a higher-confidence span; keep disjoint ones."""
    ordered = sorted(hits, key=lambda h: (-h.confidence, h.start, -(h.end - h.start)))
    kept: list[EntityHit] = []
    for h in ordered:
        if any(not (h.end <= k.start or h.start >= k.end) for k in kept):
            continue
        kept.append(h)
    return sorted(kept, key=lambda h: h.start)


_presidio_engine = None


def _presidio_hits(text: str) -> list[EntityHit]:
    global _presidio_engine
    try:
        if _presidio_engine is None:
            from presidio_analyzer import AnalyzerEngine

            _presidio_engine = AnalyzerEngine()
        results = _presidio_engine.analyze(text=text, language="en")
    except Exception:  # noqa: BLE001 - optional dependency, degrade to regex only
        return []
    return [
        EntityHit(
            entity_type=r.entity_type,
            start=r.start,
            end=r.end,
            value_hash=salted_hash(text[r.start : r.end]),
            confidence=float(r.score),
        )
        for r in results
        if r.score >= 0.5
    ]


class PIIDetector(DetectorBase):
    name = "pii"
    category = RiskCategory.PRIVACY
    cost_class = "floor"
    side = "both"

    async def _check(self, text: str, ctx: GuardContext) -> Verdict:
        hits = scan(text)
        if not hits:
            return self._verdict(0.0)
        score = max(h.confidence for h in hits)
        types = sorted({h.entity_type for h in hits})
        return self._verdict(
            score,
            evidence="entities=" + ",".join(types)
            + " spans=" + ",".join(f"{h.start}:{h.end}" for h in hits),
            entities=hits,
        )
