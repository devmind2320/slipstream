"""The invariant tests from §13. These are the ones that must never go red."""

from __future__ import annotations

import pytest

from slipstream.contracts import Action, Band, RiskCategory
from slipstream.gateway.pipeline import SlipstreamRequest
from slipstream.gateway.scheduler import policy_for_arm
from slipstream.guards.registry import build_registry
from slipstream.policy.loader import PolicyLoader
from slipstream.risk.estimator import RiskEstimator

from fixtures_data import PII_CONTEXT, PII_PROMPT


# --------------------------------------------------------------------------
# INV-1: the unconditional floor
# --------------------------------------------------------------------------
@pytest.mark.asyncio
@pytest.mark.parametrize("use_case", ["customer_support", "internal_knowledge"])
async def test_inv1_outbound_pii_masked_even_when_band_forced_low(gateway, use_case):
    """Force the cheapest possible policy and prove the leak path stays shut."""
    req = SlipstreamRequest(
        prompt=PII_PROMPT,
        use_case=use_case,
        retrieval_context=list(PII_CONTEXT),
        fixture_id="pii",
        band_override=Band.LOW,
        use_cache=False,
    )
    emitted = []
    async for ev in gateway.run(req):
        if ev.type == "chunk":
            emitted.append(ev.text)
    text = "".join(emitted)
    outcome = gateway.last_outcome

    assert outcome.record.policy.band is Band.LOW, "band override did not take effect"
    assert outcome.result.masked_before_emission
    assert "priya.sharma@example.com" not in text
    assert "ABCDE1234F" not in text
    assert "918273645012" not in text
    assert "[EMAIL]" in text
    assert outcome.result.leaked_chunks == 0


@pytest.mark.asyncio
async def test_inv1_floor_detectors_present_in_every_band(loader, registry):
    """Floor detectors are in the selected set for every band of every pack."""
    for use_case in loader.available():
        pack = loader.load(use_case)
        for band in Band:
            selected = {d.name for d in registry.select(pack.detectors_for(band))}
            assert {"pii", "injection"} <= selected, (use_case, band)


@pytest.mark.asyncio
async def test_inv1_floor_survives_degraded_mode(registry):
    """Degraded mode drops everything EXCEPT the floor."""
    degraded = {d.name for d in registry.select(["groundedness", "bias"], degraded=True)}
    assert degraded == {"pii", "injection"}


@pytest.mark.asyncio
async def test_inv1_floor_ignores_the_guard_budget(registry):
    """A zero-ish budget times out cheap detectors but never a floor one."""
    from slipstream.contracts import GuardContext

    ctx = GuardContext(retrieval_context=["some source text about a topic"])
    detectors = registry.select(["groundedness"], side="output")
    verdicts = await registry.run(
        detectors, "The account number is 918273645012.", ctx, budget_ms=1
    )
    by_name = {v.detector: v for v in verdicts}
    assert not by_name["pii"].abstained
    assert by_name["pii"].score > 0.5


# --------------------------------------------------------------------------
# INV-2: asymmetric misclassification
# --------------------------------------------------------------------------
def test_inv2_uncertainty_margin_rounds_up(loader):
    est = RiskEstimator(loader)
    pack = loader.load("internal_knowledge")
    high = pack.band_thresholds[Band.HIGH]
    med = pack.band_thresholds[Band.MEDIUM]
    margin = pack.uncertainty_margin

    # Just inside the margin below HIGH -> HIGH, never MEDIUM.
    band, reason = est._select_band(high - margin / 2, confidence=0.9, pack=pack)
    assert band is Band.HIGH, reason
    assert "INV2" in reason

    # Just inside the margin below MEDIUM -> MEDIUM, never LOW.
    band, reason = est._select_band(med - margin / 2, confidence=0.9, pack=pack)
    assert band is Band.MEDIUM, reason
    assert "INV2" in reason


def test_inv2_never_rounds_down(loader):
    """Sweep the whole score range: the band must be monotone non-decreasing and
    never below what the raw thresholds alone would give."""
    est = RiskEstimator(loader)
    pack = loader.load("internal_knowledge")
    from slipstream.contracts import BAND_ORDER

    previous = -1
    for i in range(0, 101):
        score = i / 100.0
        band, _ = est._select_band(score, confidence=0.9, pack=pack)
        raw = (
            Band.HIGH
            if score >= pack.band_thresholds[Band.HIGH]
            else Band.MEDIUM
            if score >= pack.band_thresholds[Band.MEDIUM]
            else Band.LOW
        )
        assert BAND_ORDER[band] >= BAND_ORDER[raw], f"rounded DOWN at {score}"
        assert BAND_ORDER[band] >= previous, f"non-monotone at {score}"
        previous = BAND_ORDER[band]


def test_inv2_low_confidence_rounds_up(loader):
    est = RiskEstimator(loader)
    pack = loader.load("internal_knowledge")
    band, reason = est._select_band(0.0, confidence=0.2, pack=pack)
    assert band is Band.MEDIUM
    assert "low_confidence" in reason


def test_inv2_pack_min_band_is_a_floor(loader):
    est = RiskEstimator(loader)
    pack = loader.load("decision_support")
    band, reason = est._select_band(0.0, confidence=1.0, pack=pack)
    assert band is not Band.LOW, "decision_support declares min_band: medium"


# --------------------------------------------------------------------------
# INV-3: identical detectors across arms
# --------------------------------------------------------------------------
def test_inv3_arms_share_one_registry_instance(gateway):
    """The gateway holds exactly one registry, and every arm runs against it."""
    from slipstream.gateway.scheduler import Scheduler

    pack = gateway.loader.load("internal_knowledge")
    schedulers = [
        Scheduler(gateway.provider, gateway.registry, pack) for _ in range(3)
    ]
    assert len({id(s.registry) for s in schedulers}) == 1
    assert len({s.registry.fingerprint() for s in schedulers}) == 1


def test_inv3_arm_rewrite_changes_only_scheduling_fields(loader):
    """policy_for_arm must not touch detectors, masking or escalation."""
    pack = loader.load("internal_knowledge")
    est = RiskEstimator(loader)
    from slipstream.risk.estimator import EstimatorInput

    base = est.estimate(
        EstimatorInput(prompt=PII_PROMPT, use_case="internal_knowledge")
    ).policy

    scheduling = {"holdback_tokens", "sync", "guard_budget_ms", "detectors"}
    for arm in ("fixed_sync", "fixed_async", "nemo_default"):
        rewritten = policy_for_arm(arm, base, pack)
        for field in base.model_fields:
            if field in scheduling:
                continue
            assert getattr(rewritten, field) == getattr(base, field), (arm, field)
        # All baselines run the FULL (HIGH-band) detector set, so no arm can win
        # on latency by quietly checking less.
        assert set(rewritten.detectors) == set(pack.detectors_for(Band.HIGH)), arm


def test_inv3_fingerprint_is_stable_and_sensitive():
    a, b = build_registry(), build_registry()
    assert a.fingerprint() == b.fingerprint()

    from slipstream.guards.registry import DetectorRegistry

    reduced = DetectorRegistry([a.get(n) for n in a.names if n != "bias"])
    assert reduced.fingerprint() != a.fingerprint()


# --------------------------------------------------------------------------
# INV-6: append-only ledger
# --------------------------------------------------------------------------
def test_inv6_update_and_delete_are_rejected(tmp_path):
    import sqlite3

    from slipstream.contracts import DecisionRecord, RiskVector, StreamPolicy
    from slipstream.ledger.store import Ledger

    ledger = Ledger(tmp_path / "l.sqlite3")
    rec = DecisionRecord(
        record_id="a" * 32,
        risk_vector=RiskVector(scores={RiskCategory.PRIVACY: 0.9}),
        policy=StreamPolicy(band=Band.HIGH),
        action=Action.MASK,
    )
    ledger.append(rec)

    with pytest.raises(sqlite3.IntegrityError):
        ledger.conn.execute("UPDATE decisions SET action = 'allow'")
    with pytest.raises(sqlite3.IntegrityError):
        ledger.conn.execute("DELETE FROM decisions")
    ledger.close()
