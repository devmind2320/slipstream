"""Append-only, hash-chained decision ledger (INV-6, §7.9).

``record_hash = sha256(prev_hash + canonical_json(record))``. Every record points
at its predecessor, so altering any record invalidates every hash after it and
``slipstream ledger verify`` reports the first break.

Append-only is enforced by SQLite triggers, not by convention. A caller that
tries to UPDATE or DELETE gets an exception from the database. Corrections are
new records carrying ``corrects_record_id``.

SQLite with one schema string, per §3 — a migration framework here would be
machinery for a problem this prototype does not have.
"""

from __future__ import annotations

import sqlite3
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

from .. import settings
from ..contracts import DecisionRecord, canonical_json

GENESIS = "0" * 64

SCHEMA = """
PRAGMA journal_mode=WAL;

CREATE TABLE IF NOT EXISTS decisions (
    seq            INTEGER PRIMARY KEY AUTOINCREMENT,
    record_id      TEXT NOT NULL UNIQUE,
    prev_hash      TEXT NOT NULL,
    record_hash    TEXT NOT NULL,
    trace_id       TEXT,
    use_case       TEXT,
    geography      TEXT,
    conversation_id TEXT,
    turn_index     INTEGER,
    action         TEXT,
    band           TEXT,
    arm            TEXT,
    degraded_mode  INTEGER,
    abstained      INTEGER,
    corrects_record_id TEXT,
    ttft_ms        REAL,
    total_latency_ms REAL,
    guard_overhead_ms REAL,
    timestamp      TEXT NOT NULL,
    payload        TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_decisions_trace ON decisions(trace_id);
CREATE INDEX IF NOT EXISTS idx_decisions_conv ON decisions(conversation_id);

-- INV-6. Append-only is a property of the store, not a promise in a README.
CREATE TRIGGER IF NOT EXISTS decisions_no_update
BEFORE UPDATE ON decisions
BEGIN
    SELECT RAISE(ABORT, 'ledger is append-only: UPDATE rejected');
END;

CREATE TRIGGER IF NOT EXISTS decisions_no_delete
BEFORE DELETE ON decisions
BEGIN
    SELECT RAISE(ABORT, 'ledger is append-only: DELETE rejected');
END;

CREATE TABLE IF NOT EXISTS feedback (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    record_id     TEXT NOT NULL,
    reviewer      TEXT,
    label         TEXT NOT NULL,
    category      TEXT,
    note          TEXT,
    timestamp     TEXT NOT NULL
);
"""


@dataclass
class ChainBreak:
    seq: int
    record_id: str
    expected: str
    found: str
    kind: str


class Ledger:
    def __init__(self, path: Path | None = None) -> None:
        self.path = Path(path or settings.LEDGER_PATH)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(SCHEMA)
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()

    def __enter__(self) -> "Ledger":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # -- write -----------------------------------------------------------
    def head_hash(self) -> str:
        row = self.conn.execute(
            "SELECT record_hash FROM decisions ORDER BY seq DESC LIMIT 1"
        ).fetchone()
        return row["record_hash"] if row else GENESIS

    def append(self, record: DecisionRecord, *, corrects: str | None = None) -> DecisionRecord:
        record = record.model_copy(update={"prev_hash": self.head_hash()})
        if not record.record_id:
            record = record.model_copy(update={"record_id": uuid.uuid4().hex})
        record = record.model_copy(update={"record_hash": record.compute_hash()})

        self.conn.execute(
            """INSERT INTO decisions (
                record_id, prev_hash, record_hash, trace_id, use_case, geography,
                conversation_id, turn_index, action, band, arm, degraded_mode,
                abstained, corrects_record_id, ttft_ms, total_latency_ms,
                guard_overhead_ms, timestamp, payload
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                record.record_id,
                record.prev_hash,
                record.record_hash,
                record.trace_id,
                record.use_case,
                record.geography,
                record.conversation_id,
                record.turn_index,
                record.action.value,
                record.policy.band.value,
                record.arm,
                int(record.degraded_mode),
                int(record.abstained),
                corrects,
                record.ttft_ms,
                record.total_latency_ms,
                record.guard_overhead_ms,
                record.timestamp.isoformat(),
                canonical_json(record.model_dump(mode="json")),
            ),
        )
        self.conn.commit()
        return record

    def correct(self, original_record_id: str, replacement: DecisionRecord) -> DecisionRecord:
        """INV-6: corrections are new records referencing the original."""
        return self.append(replacement, corrects=original_record_id)

    def add_feedback(
        self,
        record_id: str,
        label: str,
        *,
        reviewer: str = "human",
        category: str | None = None,
        note: str | None = None,
    ) -> None:
        self.conn.execute(
            "INSERT INTO feedback (record_id, reviewer, label, category, note, timestamp)"
            " VALUES (?,?,?,?,?,?)",
            (
                record_id,
                reviewer,
                label,
                category,
                note,
                datetime.now(timezone.utc).isoformat(),
            ),
        )
        self.conn.commit()

    # -- read ------------------------------------------------------------
    def __len__(self) -> int:
        return int(self.conn.execute("SELECT COUNT(*) c FROM decisions").fetchone()["c"])

    def rows(self) -> Iterator[sqlite3.Row]:
        yield from self.conn.execute("SELECT * FROM decisions ORDER BY seq ASC")

    def recent(self, limit: int = 20) -> list[dict]:
        rows = self.conn.execute(
            "SELECT * FROM decisions ORDER BY seq DESC LIMIT ?", (limit,)
        ).fetchall()
        return [dict(r) for r in rows]

    def get(self, record_id: str) -> dict | None:
        row = self.conn.execute(
            "SELECT * FROM decisions WHERE record_id = ?", (record_id,)
        ).fetchone()
        return dict(row) if row else None

    def feedback_rows(self) -> list[dict]:
        return [
            dict(r)
            for r in self.conn.execute("SELECT * FROM feedback ORDER BY id ASC")
        ]

    # -- integrity -------------------------------------------------------
    def verify(self) -> ChainBreak | None:
        return verify_chain(self)


def verify_chain(ledger: Ledger) -> ChainBreak | None:
    """Walk the chain and report the FIRST break, or None if intact."""
    import hashlib
    import json

    prev = GENESIS
    for row in ledger.rows():
        if row["prev_hash"] != prev:
            return ChainBreak(
                seq=row["seq"],
                record_id=row["record_id"],
                expected=prev,
                found=row["prev_hash"],
                kind="broken_link",
            )
        payload = json.loads(row["payload"])
        payload.pop("record_hash", None)
        recomputed = hashlib.sha256(
            (row["prev_hash"] + canonical_json(payload)).encode("utf-8")
        ).hexdigest()
        if recomputed != row["record_hash"]:
            return ChainBreak(
                seq=row["seq"],
                record_id=row["record_id"],
                expected=recomputed,
                found=row["record_hash"],
                kind="payload_tampered",
            )
        prev = row["record_hash"]
    return None
