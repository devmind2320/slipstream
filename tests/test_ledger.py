"""Ledger integrity, OTel attributes, cache gating and policy hot reload."""

from __future__ import annotations

import json
import time

import pytest

from slipstream.contracts import (
    Action,
    Band,
    DecisionRecord,
    RiskCategory,
    RiskVector,
    StreamPolicy,
    canonical_json,
)
from slipstream.ledger.store import GENESIS, Ledger, verify_chain
from slipstream.routing.cache import SemanticCache, entity_signature


def record(action=Action.ALLOW, rid="") -> DecisionRecord:
    import uuid

    return DecisionRecord(
        record_id=rid or uuid.uuid4().hex,
        risk_vector=RiskVector(scores={RiskCategory.PRIVACY: 0.4}),
        policy=StreamPolicy(band=Band.MEDIUM, detectors=["pii", "injection"]),
        action=action,
        use_case="internal_knowledge",
    )


@pytest.fixture
def ledger(tmp_path):
    lg = Ledger(tmp_path / "ledger.sqlite3")
    yield lg
    lg.close()


# --------------------------------------------------------------------------
def test_chain_links_and_verifies(ledger):
    first = ledger.append(record())
    assert first.prev_hash == GENESIS
    prev = first.record_hash
    for _ in range(6):
        r = ledger.append(record())
        assert r.prev_hash == prev
        assert r.record_hash == r.compute_hash()
        prev = r.record_hash
    assert verify_chain(ledger) is None
    assert len(ledger) == 7


def test_tampering_with_a_payload_is_detected(ledger):
    for _ in range(5):
        ledger.append(record())
    rows = list(ledger.rows())
    target = rows[2]

    payload = json.loads(target["payload"])
    payload["action"] = "allow" if payload["action"] != "allow" else "block"
    # Bypass the append-only triggers the way an attacker with file access would:
    # rewrite the row directly, leaving the stored record_hash intact.
    ledger.conn.execute("DROP TRIGGER decisions_no_update")
    ledger.conn.execute(
        "UPDATE decisions SET payload = ? WHERE seq = ?",
        (canonical_json(payload), target["seq"]),
    )
    ledger.conn.commit()

    brk = verify_chain(ledger)
    assert brk is not None, "tampering went undetected"
    assert brk.seq == target["seq"]
    assert brk.kind == "payload_tampered"


def test_tampering_with_a_link_is_detected(ledger):
    for _ in range(4):
        ledger.append(record())
    rows = list(ledger.rows())
    ledger.conn.execute("DROP TRIGGER decisions_no_update")
    ledger.conn.execute(
        "UPDATE decisions SET prev_hash = ? WHERE seq = ?",
        ("f" * 64, rows[2]["seq"]),
    )
    ledger.conn.commit()
    brk = verify_chain(ledger)
    assert brk is not None and brk.kind == "broken_link"
    assert brk.seq == rows[2]["seq"]


def test_verify_reports_the_FIRST_break(ledger):
    for _ in range(6):
        ledger.append(record())
    rows = list(ledger.rows())
    ledger.conn.execute("DROP TRIGGER decisions_no_update")
    for seq in (rows[4]["seq"], rows[2]["seq"]):
        ledger.conn.execute(
            "UPDATE decisions SET prev_hash = ? WHERE seq = ?", ("e" * 64, seq)
        )
    ledger.conn.commit()
    brk = verify_chain(ledger)
    assert brk.seq == rows[2]["seq"], "reported a later break than the first"


def test_corrections_are_new_records_referencing_the_original(ledger):
    original = ledger.append(record(Action.ALLOW))
    correction = ledger.correct(original.record_id, record(Action.ESCALATE))
    assert verify_chain(ledger) is None
    assert len(ledger) == 2
    row = ledger.get(correction.record_id)
    assert row["corrects_record_id"] == original.record_id
    # The original is untouched.
    assert ledger.get(original.record_id)["action"] == "allow"


def test_ledger_never_stores_raw_pii(ledger):
    from slipstream.contracts import EntityHit, Verdict

    rec = record()
    rec = rec.model_copy(
        update={
            "entities_masked": ["EMAIL", "PAN"],
            "verdicts": [
                Verdict(
                    detector="pii",
                    category=RiskCategory.PRIVACY,
                    score=0.98,
                    latency_ms=2.0,
                    evidence="entities=EMAIL,PAN spans=8:32",
                    entities=[
                        EntityHit(
                            entity_type="EMAIL",
                            start=8,
                            end=32,
                            value_hash="abc123abc123abc1",
                        )
                    ],
                )
            ],
        }
    )
    stored = ledger.append(rec)
    blob = ledger.get(stored.record_id)["payload"]
    assert "priya.sharma@example.com" not in blob
    assert "ABCDE1234F" not in blob
    assert "EMAIL" in blob  # the TYPE is recorded
    assert "abc123abc123abc1" in blob  # the salted hash is recorded


# --------------------------------------------------------------------------
def test_otel_span_attributes_match_the_pinned_set(tmp_path, monkeypatch):
    from slipstream import settings
    from slipstream.ledger import otel

    path = tmp_path / "spans.jsonl"
    monkeypatch.setattr(settings, "OTEL_FILE", path)
    otel.shutdown_telemetry()
    otel.emit_span(record(), pct_guard_first=0.75)
    otel.shutdown_telemetry()

    lines = [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x]
    assert lines, "no span was exported"
    attrs = lines[-1]["attributes"]
    assert set(attrs) == set(otel.ATTRIBUTE_SET), (
        set(attrs) ^ set(otel.ATTRIBUTE_SET)
    )
    assert attrs["gen_ai.operation.name"] == "chat"
    assert attrs["slipstream.pct_guard_finished_first"] == 0.75


# --------------------------------------------------------------------------
def test_entity_gated_cache_rejects_a_near_miss():
    """The whole point of the gate: high similarity, different identifier."""
    cache = SemanticCache(similarity_threshold=0.80)
    a = "What is the balance on account 918273645012 for this customer?"
    b = "What is the balance on account 918273645099 for this customer?"

    cache.put(a, "customer_support", "the balance is X")
    value, outcome = cache.get(b, "customer_support")
    assert value is None
    assert outcome == "gated_miss", outcome
    assert cache.stats.gated_misses == 1


def test_cache_serves_an_exact_repeat():
    cache = SemanticCache()
    cache.put("what is the refund window", "customer_support", "five days")
    value, outcome = cache.get("what is the refund window", "customer_support")
    assert value == "five days" and outcome == "exact"


def test_cache_is_scoped_per_use_case():
    cache = SemanticCache()
    cache.put("same question", "customer_support", "A")
    value, outcome = cache.get("same question", "decision_support")
    assert value is None


def test_entity_signature_covers_names_and_figures():
    sig_a = entity_signature("Pay 42,50,000 to Priya Sharma at account 918273645012")
    sig_b = entity_signature("Pay 42,50,000 to Priya Sharma at account 918273645013")
    assert sig_a != sig_b
    assert entity_signature("no identifiers here") == frozenset()


# --------------------------------------------------------------------------
def test_policy_pack_hot_reload(tmp_path):
    from slipstream.policy.loader import PolicyLoader

    pack_file = tmp_path / "temp_case.yaml"
    pack_file.write_text(
        "use_case: temp_case\nlatency_budget_ms: 1000\n"
        "bands:\n  low: { holdback_tokens: 8, sync: false, guard_budget_ms: 50 }\n"
        "detectors:\n  floor: [pii, injection]\n",
        encoding="utf-8",
    )
    loader = PolicyLoader(tmp_path)
    assert loader.load("temp_case").latency_budget_ms == 1000

    time.sleep(0.01)
    pack_file.write_text(
        "use_case: temp_case\nlatency_budget_ms: 4321\n"
        "bands:\n  low: { holdback_tokens: 8, sync: false, guard_budget_ms: 50 }\n"
        "detectors:\n  floor: [pii, injection]\n",
        encoding="utf-8",
    )
    assert loader.load("temp_case").latency_budget_ms == 4321, "hot reload did not fire"


def test_all_three_shipped_packs_load():
    from slipstream.policy.loader import PolicyLoader

    loader = PolicyLoader()
    assert set(loader.available()) == {
        "customer_support",
        "internal_knowledge",
        "decision_support",
    }
    for name, pack in loader.load_all().items():
        assert pack.use_case == name
        assert "pii" in pack.detectors["floor"]
        assert "injection" in pack.detectors["floor"]
        assert pack.latency_budget_ms > 0


def test_decision_support_never_runs_low():
    from slipstream.policy.loader import PolicyLoader

    pack = PolicyLoader().load("decision_support")
    assert pack.min_band is Band.MEDIUM
    assert Band.LOW not in pack.bands
    # And asking for a LOW config must NOT return something laxer than MEDIUM.
    assert pack.band_config(Band.LOW).holdback_tokens >= pack.band_config(
        Band.MEDIUM
    ).holdback_tokens
