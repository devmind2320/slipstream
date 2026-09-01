"""Determinism and end-to-end reproducibility (§13, INV-5).

The contract this file pins down: **decisions are deterministic, timings are
not.** Two seeded runs must produce byte-identical corpora and byte-identical
decision digests. Wall-clock latency depends on the machine and on scheduler
jitter, so asserting equality on it would give a test that passes by luck; it is
excluded from the digest and that exclusion is documented in the report.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import asdict

import pytest

from slipstream.eval.datasets import generate as gen
from slipstream.eval.harness import deterministic_digest, run_arm, scoring_pass
from slipstream.gateway.providers.replay import ReplayProvider
from slipstream.guards.registry import build_registry
from slipstream.policy.loader import PolicyLoader


def test_corpus_is_byte_identical_under_the_same_seed():
    a = gen.generate(seed=4242, scale=0.03)
    b = gen.generate(seed=4242, scale=0.03)
    assert [asdict(x) for x in a] == [asdict(x) for x in b]


def test_corpus_differs_under_a_different_seed():
    a = gen.generate(seed=4242, scale=0.03)
    b = gen.generate(seed=99, scale=0.03)
    assert [x.prompt for x in a] != [x.prompt for x in b]


def test_labels_are_sets_and_the_overlap_slice_carries_both():
    items = gen.generate(seed=4242, scale=0.05)
    overlap = [i for i in items if i.slice == "overlap_person_fabrication"]
    assert overlap
    for i in overlap:
        assert set(i.labels) == {"privacy", "hallucination"}, i.labels


def test_unverifiable_slice_is_labelled_benign_and_expects_abstention():
    """Scoring these as violations would reward the failure mode §9 warns about."""
    items = gen.generate(seed=4242, scale=0.05)
    unverifiable = [i for i in items if i.slice == "unverifiable_no_context"]
    assert unverifiable
    for i in unverifiable:
        assert i.labels == []
        assert i.expects_abstention is True
        assert i.retrieval_context == []


def test_split_is_stratified_and_keeps_conversations_whole():
    items = gen.generate(seed=4242, scale=0.25)
    by_slice: dict[str, set[str]] = {}
    for i in items:
        by_slice.setdefault(i.slice, set()).add(i.split)
    for slice_name, splits in by_slice.items():
        if slice_name == "multi_turn":
            continue
        assert splits == {"calibration", "test"}, (slice_name, splits)

    conv_splits: dict[str, set[str]] = {}
    for i in items:
        if i.conversation_id:
            conv_splits.setdefault(i.conversation_id, set()).add(i.split)
    assert all(len(s) == 1 for s in conv_splits.values()), "a session was split"


def test_every_pack_has_positives_for_the_categories_it_enforces():
    """A (pack, category) pair with no positives cannot be conformally fitted,
    so the corpus must cover the ones each pack actually acts on."""
    items = gen.generate(seed=4242, scale=1.0)
    seen: dict[tuple[str, str], int] = {}
    for i in items:
        for lab in i.labels:
            seen[(i.use_case, lab)] = seen.get((i.use_case, lab), 0) + 1
    for use_case in ("customer_support", "internal_knowledge", "decision_support"):
        for cat in ("privacy", "hallucination"):
            assert seen.get((use_case, cat), 0) >= 20, (
                f"{use_case}/{cat} has {seen.get((use_case, cat), 0)} positives"
            )


@pytest.mark.asyncio
async def test_two_seeded_runs_produce_identical_decision_digests():
    """The §13 determinism test."""
    items = gen.generate(seed=4242, scale=0.03)
    test_items = [i for i in items if i.split == "test"]
    assert test_items

    async def one_run() -> dict[str, str]:
        provider = ReplayProvider([i.to_fixture() for i in items], time_scale=0.2)
        registry = build_registry()
        loader = PolicyLoader()
        scores = await scoring_pass(registry, provider, loader, items, 8)
        by_arm = {}
        for arm in ("fixed_sync", "fixed_async", "adaptive"):
            by_arm[arm] = await run_arm(
                arm,
                test_items,
                registry=registry,
                provider=provider,
                loader=loader,
                scores=scores,
                concurrency=8,
                # The two wall-clock-dependent inputs to a decision, both
                # disabled so the run is bit-for-bit reproducible. Their effect
                # is measured separately in the eval.
                backpressure_k=0,
                enforce_budget=False,
            )
        return deterministic_digest(by_arm)

    first = await one_run()
    second = await one_run()
    assert first == second, f"non-deterministic decisions:\n{first}\n{second}"
    # With degradation off, nothing may be excluded from the digest.
    for arm in ("fixed_sync", "fixed_async", "adaptive"):
        assert first[f"{arm}__n_excluded_load_dependent"] == "0"
        assert int(first[f"{arm}__n_items_digested"]) == len(test_items)
    # And the arms must not be trivially identical, or the digest proves nothing.
    digests = {first[a] for a in ("fixed_sync", "fixed_async", "adaptive")}
    assert len(digests) > 1, "all arms produced the same decisions"


@pytest.mark.asyncio
async def test_replay_provider_is_deterministic():
    from slipstream.gateway.providers.base import GenerationRequest

    provider = ReplayProvider(
        [{"fixture_id": "f", "prompt": "p", "response": "one two three four five."}],
        time_scale=0.1,
    )

    async def collect():
        return [t.text async for t in provider.stream(GenerationRequest(prompt="p"))]

    assert await collect() == await collect()
    assert "".join(await collect()) == "one two three four five."


@pytest.mark.asyncio
async def test_replay_provider_raises_a_useful_error_for_a_missing_fixture():
    from slipstream.gateway.providers.base import GenerationRequest

    provider = ReplayProvider([])
    with pytest.raises(KeyError, match="make data"):
        async for _ in provider.stream(GenerationRequest(prompt="nothing here")):
            pass


@pytest.mark.asyncio
async def test_offline_end_to_end_needs_no_network_or_key(monkeypatch):
    """INV-5, enforced rather than asserted: any socket use fails the test."""
    import socket

    from slipstream.demo.scenario import DEMO_FIXTURES, get_scenario
    from slipstream.gateway.pipeline import SlipstreamGateway, SlipstreamRequest

    def no_network(*args, **kwargs):
        raise AssertionError("the offline path attempted a network connection")

    monkeypatch.setattr(socket.socket, "connect", no_network)
    monkeypatch.delenv("SLIPSTREAM_OPENAI_API_KEY", raising=False)

    sc = get_scenario("compensation")
    gateway = SlipstreamGateway(
        ReplayProvider(DEMO_FIXTURES, time_scale=0.1),
        registry=build_registry(),
        loader=PolicyLoader(),
        emit_telemetry=False,
    )
    outcome = await gateway.collect(
        SlipstreamRequest(
            prompt=sc["prompt"],
            use_case=sc["use_case"],
            retrieval_context=list(sc["retrieval_context"]),
            fixture_id=sc["fixture_id"],
            arm="adaptive",
            use_cache=False,
        )
    )
    assert outcome.record.entities_masked
    assert outcome.result.leaked_chunks == 0


def test_results_payload_is_json_serialisable_and_sorted():
    """INV-4 plumbing: whatever the harness writes must round-trip."""
    from slipstream.eval.metrics import ItemResult, compute_arm_metrics

    results = [
        ItemResult(
            item_id=f"i{i}",
            slice="benign_customer_support",
            use_case="customer_support",
            arm="adaptive",
            band="low",
            labels=[],
            predicted=[],
            action="allow",
            category_scores={"privacy": 0.1, "hallucination": 0.2},
        )
        for i in range(5)
    ]
    payload = compute_arm_metrics(results, {"customer_support": 0.02})
    # allow_nan=False is the point: bare NaN is not valid JSON, and results.json
    # has to be a file a reviewer can open with any parser.
    text = json.dumps(payload, sort_keys=True, allow_nan=False)
    assert json.loads(text) == payload
    assert "NaN" not in text and "Infinity" not in text
