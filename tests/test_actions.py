"""Action matrix tests (§8) — especially max-severity and abstention routing."""

from __future__ import annotations

import pytest

from slipstream.actions.matrix import (
    UNCERTAIN_NOTICE,
    UNVERIFIABLE_NOTICE,
    decide,
)
from slipstream.contracts import (
    Action,
    Band,
    EntityHit,
    RiskCategory,
    Verdict,
    max_severity,
)
from slipstream.policy.loader import PolicyLoader


def v(category, score, *, detector=None, abstained=False, entities=None) -> Verdict:
    return Verdict(
        detector=detector or category.value,
        category=category,
        score=score,
        latency_ms=1.0,
        abstained=abstained,
        entities=entities or [],
    )


def entity(t="EMAIL") -> EntityHit:
    return EntityHit(entity_type=t, start=0, end=5, value_hash="deadbeefdeadbeef")


@pytest.fixture
def packs():
    loader = PolicyLoader()
    return {n: loader.load(n) for n in loader.available()}


# --------------------------------------------------------------------------
def test_injection_blocks(packs):
    d = decide([v(RiskCategory.INJECTION, 0.95)], packs["customer_support"])
    assert d.action is Action.BLOCK
    assert d.fallback_text


def test_outbound_pii_masks_when_masking_enabled(packs):
    d = decide(
        [v(RiskCategory.PRIVACY, 0.95, entities=[entity()])],
        packs["customer_support"],
        masking_enabled=True,
    )
    assert d.action is Action.MASK
    assert d.entity_types == ["EMAIL"]


def test_outbound_pii_blocks_when_masking_required_but_disabled(packs):
    pack = packs["customer_support"]
    assert pack.require_masking
    d = decide(
        [v(RiskCategory.PRIVACY, 0.95, entities=[entity()])],
        pack,
        masking_enabled=False,
    )
    assert d.action is Action.BLOCK


def test_inbound_pii_is_not_the_leak_path(packs):
    d = decide(
        [v(RiskCategory.PRIVACY, 0.95, entities=[entity()])],
        packs["customer_support"],
        is_prompt_side=True,
    )
    assert d.action is Action.ANNOTATE


def test_groundedness_escalates_in_a_regulated_pack(packs):
    d = decide(
        [v(RiskCategory.HALLUCINATION, 0.95, detector="groundedness")],
        packs["decision_support"],
        verifiable=True,
    )
    assert d.action is Action.ESCALATE


def test_groundedness_regenerates_in_an_unregulated_pack(packs):
    d = decide(
        [v(RiskCategory.HALLUCINATION, 0.95, detector="groundedness")],
        packs["internal_knowledge"],
        verifiable=True,
    )
    assert d.action is Action.REGENERATE


def test_unverifiable_annotates_with_an_explicit_notice(packs):
    d = decide(
        [v(RiskCategory.HALLUCINATION, 0.95, detector="groundedness")],
        packs["internal_knowledge"],
        verifiable=False,
    )
    assert d.action is Action.ANNOTATE
    assert UNVERIFIABLE_NOTICE in d.annotations


def test_toxicity_blocks_with_a_compliant_fallback(packs):
    d = decide([v(RiskCategory.TOXICITY, 0.9)], packs["customer_support"])
    assert d.action is Action.BLOCK
    assert "content policy" in (d.fallback_text or "")


def test_bias_escalates_in_a_decision_context(packs):
    d = decide([v(RiskCategory.BIAS, 0.9)], packs["decision_support"])
    assert d.action is Action.ESCALATE


def test_clean_allows(packs):
    d = decide([v(RiskCategory.PRIVACY, 0.0), v(RiskCategory.TOXICITY, 0.0)],
               packs["customer_support"])
    assert d.action is Action.ALLOW
    assert not d.triggered


# --------------------------------------------------------------------------
# §9: abstention routing
# --------------------------------------------------------------------------
def test_abstention_escalates_in_a_regulated_pack(packs):
    d = decide(
        [v(RiskCategory.HALLUCINATION, 0.0, detector="groundedness", abstained=True)],
        packs["decision_support"],
    )
    assert d.action is Action.ESCALATE
    assert RiskCategory.HALLUCINATION in d.abstained
    assert RiskCategory.HALLUCINATION not in d.triggered


def test_abstention_annotates_in_a_low_stakes_pack(packs):
    d = decide(
        [v(RiskCategory.HALLUCINATION, 0.0, detector="groundedness", abstained=True)],
        packs["customer_support"],
    )
    assert d.action is Action.ANNOTATE
    assert UNVERIFIABLE_NOTICE in d.annotations


def test_abstention_is_never_counted_as_a_violation(packs):
    d = decide(
        [v(RiskCategory.PRIVACY, 0.99, abstained=True)], packs["customer_support"]
    )
    assert RiskCategory.PRIVACY not in d.triggered
    assert d.action is Action.ALLOW


# --------------------------------------------------------------------------
# §9: max severity across triggered categories, not first match
# --------------------------------------------------------------------------
def test_max_severity_not_first_match(packs):
    """Ordering the verdicts differently must not change the outcome."""
    verdicts = [
        v(RiskCategory.HALLUCINATION, 0.99, detector="groundedness"),  # -> regenerate
        v(RiskCategory.INJECTION, 0.99),                               # -> block
        v(RiskCategory.PRIVACY, 0.99, entities=[entity()]),            # -> mask
    ]
    forward = decide(list(verdicts), packs["internal_knowledge"], verifiable=True)
    backward = decide(list(reversed(verdicts)), packs["internal_knowledge"], verifiable=True)
    assert forward.action is Action.BLOCK
    assert backward.action is Action.BLOCK
    assert forward.action == backward.action


def test_masking_entities_survive_a_higher_severity_action(packs):
    """The bug this test exists for: a chunk that trips both PII and
    groundedness must still have its entities collected for masking, even though
    REGENERATE outranks MASK on the severity ladder."""
    d = decide(
        [
            v(RiskCategory.PRIVACY, 0.99, entities=[entity("PAN")]),
            v(RiskCategory.HALLUCINATION, 0.99, detector="groundedness"),
        ],
        packs["internal_knowledge"],
        verifiable=True,
    )
    assert d.action is Action.REGENERATE
    assert d.entities_to_mask, "entities were dropped when a higher action won"
    assert d.entity_types == ["PAN"]


def test_max_severity_helper():
    assert max_severity([]) is Action.ALLOW
    assert max_severity([Action.ANNOTATE, Action.BLOCK, Action.MASK]) is Action.BLOCK
    assert max_severity([Action.ALLOW, Action.ANNOTATE]) is Action.ANNOTATE


# --------------------------------------------------------------------------
# Uncertainty margin
# --------------------------------------------------------------------------
def test_uncertainty_margin_annotates_and_marks_for_feedback(packs):
    pack = packs["internal_knowledge"]
    thr = pack.threshold(RiskCategory.TOXICITY)
    d = decide([v(RiskCategory.TOXICITY, thr - pack.uncertainty_margin / 2)], pack)
    assert d.action is Action.ANNOTATE
    assert d.near_threshold and d.feedback_worthy
    assert UNCERTAIN_NOTICE in d.annotations


def test_escalation_degrades_in_low_band_of_an_unregulated_pack(packs):
    d = decide(
        [v(RiskCategory.HALLUCINATION, 0.0, detector="groundedness", abstained=True)],
        packs["customer_support"],
        band=Band.LOW,
    )
    assert d.action is Action.ANNOTATE
