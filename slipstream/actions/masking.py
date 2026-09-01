"""Entity masking.

Masking happens on the held-back chunk *before* it is released, which is the
whole point of hold-back streaming: in the fixed_async arm the same entity is
already on the user's screen when the verdict arrives.

The masked text carries the entity TYPE so the response stays readable —
"transfer to [ACCOUNT]" is a usable sentence, "transfer to ████" is not. Values
never appear in the output, in the log, or in the ledger.
"""

from __future__ import annotations

from ..contracts import EntityHit
from ..guards.pii import scan

PLACEHOLDER = "[{}]"


def mask_entities(text: str, entities: list[EntityHit]) -> tuple[str, list[str]]:
    """Replace each entity span with its type placeholder.

    Applied right-to-left so earlier offsets stay valid.
    """
    if not entities:
        return text, []
    ordered = sorted(entities, key=lambda e: e.start, reverse=True)
    out = text
    types: list[str] = []
    for e in ordered:
        if not (0 <= e.start < e.end <= len(out)):
            continue
        out = out[: e.start] + PLACEHOLDER.format(e.entity_type) + out[e.end :]
        types.append(e.entity_type)
    return out, sorted(set(types))


def mask_text(text: str) -> tuple[str, list[str]]:
    """Scan and mask in one call. Used where no verdict is at hand."""
    return mask_entities(text, scan(text))
