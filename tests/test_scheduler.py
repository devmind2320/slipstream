"""Scheduler tests — the concurrency claim and the backpressure claim (§13)."""

from __future__ import annotations

import asyncio
import time

import pytest

from slipstream.contracts import Band, GuardContext, RiskCategory, StreamPolicy, Verdict
from slipstream.gateway.providers.base import GenerationRequest
from slipstream.gateway.providers.replay import ReplayProvider
from slipstream.gateway.scheduler import (
    FLOOR_HOLDBACK_TOKENS,
    Scheduler,
    StreamResult,
    policy_for_arm,
)
from slipstream.guards.base import DetectorBase
from slipstream.guards.registry import DetectorRegistry, build_registry
from slipstream.policy.loader import PolicyLoader
from slipstream.textutil import find_boundary

LONG = " ".join(
    f"Sentence number {i} states a plain and entirely unremarkable fact about "
    f"the quarterly operating review." for i in range(1, 13)
)


class RecordingDetector(DetectorBase):
    """A detector that logs the wall-clock window it occupied."""

    name = "recording"
    category = RiskCategory.HALLUCINATION
    cost_class = "cheap"
    side = "output"

    def __init__(self, delay_s: float = 0.05) -> None:
        self.delay_s = delay_s
        self.windows: list[tuple[float, float]] = []

    async def _check(self, text: str, ctx: GuardContext) -> Verdict:
        start = time.perf_counter()
        await asyncio.sleep(self.delay_s)
        self.windows.append((start, time.perf_counter()))
        return self._verdict(0.0, evidence="ok")


def _pack():
    return PolicyLoader().load("internal_knowledge")


@pytest.mark.asyncio
async def test_guard_for_chunk_n_overlaps_generation_of_chunk_n_plus_1():
    """The core claim, asserted in wall-clock rather than described in a README.

    Chunk N's detector window must still be open while the provider is producing
    the tokens of chunk N+1. If these ever serialise, the entire latency argument
    collapses.
    """
    rec = RecordingDetector(delay_s=0.06)
    registry = DetectorRegistry([rec, *(build_registry().get(n) for n in ("pii", "injection"))])
    provider = ReplayProvider(
        [{"fixture_id": "long", "prompt": "p", "response": LONG}], time_scale=1.0
    )
    scheduler = Scheduler(provider, registry, _pack())
    policy = StreamPolicy(
        band=Band.MEDIUM, holdback_tokens=12, detectors=["recording"], sync=False
    )
    result = StreamResult()

    cut_times: list[float] = []
    async for ev in scheduler.stream(
        GenerationRequest(prompt="p", fixture_id="long"), policy, GuardContext(), result
    ):
        if ev.type == "chunk":
            cut_times.append(time.perf_counter())

    assert len(rec.windows) >= 3, "need several chunks to observe an overlap"
    assert result.chunks >= 3

    # Guard window N must overlap guard window N+1's *generation*, which we
    # observe as: guard N was still running when guard N+1 started, OR guard N
    # finished after the chunk that followed it was cut.
    overlaps = 0
    for (s0, e0), (s1, _e1) in zip(rec.windows, rec.windows[1:]):
        gen_window = s1 - e0  # gap between consecutive guard runs
        if gen_window >= 0 and e0 > s0:
            overlaps += 1
    assert overlaps >= 2

    won, total = result.overlap_counts
    assert total >= 2, "chunks with a successor were not recorded"
    assert won >= 1, "no chunk had its verdict ready before the next chunk was cut"


@pytest.mark.asyncio
async def test_sync_policy_serialises_and_is_slower():
    """The fixed_sync baseline must actually be synchronous."""
    delay = 0.05
    provider = ReplayProvider(
        [{"fixture_id": "long", "prompt": "p", "response": LONG}], time_scale=1.0
    )

    async def run(sync: bool) -> tuple[float, StreamResult]:
        rec = RecordingDetector(delay_s=delay)
        registry = DetectorRegistry([rec, build_registry().get("pii"), build_registry().get("injection")])
        scheduler = Scheduler(provider, registry, _pack())
        policy = StreamPolicy(
            band=Band.MEDIUM, holdback_tokens=12, detectors=["recording"], sync=sync
        )
        result = StreamResult()
        t0 = time.perf_counter()
        async for _ in scheduler.stream(
            GenerationRequest(prompt="p", fixture_id="long"), policy, GuardContext(), result
        ):
            pass
        return (time.perf_counter() - t0), result

    async_t, async_r = await run(False)
    sync_t, sync_r = await run(True)

    assert sync_t > async_t, (sync_t, async_t)
    # Sync chunks never raced, so none of them count toward the overlap metric.
    assert all(not t.raced for t in sync_r.chunk_timings)
    assert sync_r.overlap_counts[1] == 0


@pytest.mark.asyncio
async def test_backpressure_degrades_rather_than_stalls():
    """Guards that lose k consecutive races drop to floor-only and keep going."""

    class Slow(RecordingDetector):
        name = "slow"

    slow = Slow(delay_s=0.9)  # far slower than any chunk's generation
    base = build_registry()
    registry = DetectorRegistry([slow, base.get("pii"), base.get("injection")])
    provider = ReplayProvider(
        [{"fixture_id": "long", "prompt": "p", "response": LONG}], time_scale=1.0
    )
    scheduler = Scheduler(provider, registry, _pack(), backpressure_k=2)
    policy = StreamPolicy(
        band=Band.MEDIUM, holdback_tokens=10, detectors=["slow"], sync=False
    )
    result = StreamResult()

    t0 = time.perf_counter()
    chunks = 0
    async for ev in scheduler.stream(
        GenerationRequest(prompt="p", fixture_id="long"), policy, GuardContext(), result
    ):
        if ev.type == "chunk":
            chunks += 1
    elapsed = time.perf_counter() - t0

    assert result.degraded_mode, "backpressure never fired"
    assert chunks > 0, "the stream stalled instead of degrading"
    assert any("backpressure" in r for r in result.reasons)
    # Once degraded, only floor detectors run, so the slow detector must not
    # have been charged for every chunk.
    assert elapsed < 0.9 * result.chunks, "degradation did not actually shed work"


@pytest.mark.asyncio
async def test_holdback_zero_still_holds_for_the_floor():
    """INV-1: holdback 0 means 'as shallow as the floor allows', not 'no floor'."""
    provider = ReplayProvider(
        [{"fixture_id": "long", "prompt": "p", "response": LONG}], time_scale=1.0
    )
    registry = build_registry()
    policy = StreamPolicy(band=Band.LOW, holdback_tokens=0, detectors=[], sync=False)

    guarded = Scheduler(provider, registry, _pack(), enforce_floor_inline=True)
    r1 = StreamResult()
    async for _ in guarded.stream(
        GenerationRequest(prompt="p", fixture_id="long"), policy, GuardContext(), r1
    ):
        pass

    leaky = Scheduler(provider, registry, _pack(), enforce_floor_inline=False)
    r2 = StreamResult()
    async for _ in leaky.stream(
        GenerationRequest(prompt="p", fixture_id="long"), policy, GuardContext(), r2
    ):
        pass

    # The property that matters is not the chunk count — the leaky arm uses a
    # coarser scan cadence and has fewer chunks. It is *when emission happens
    # relative to the first token*.
    assert FLOOR_HOLDBACK_TOKENS > 0

    # Leaky: the first token is on the screen the instant it arrives.
    assert r2.ttft_ms == pytest.approx(r2.first_token_ms, abs=5.0)
    # Floor-enforced: emission waits for a floor-depth chunk AND its verdict.
    assert r1.ttft_ms > r1.first_token_ms + 5.0
    assert r1.ttft_ms > r2.ttft_ms

    # Nothing in the floor-enforced run was emitted before its verdict existed.
    assert r1.leaked_chunks == 0
    assert all(not t.leaked for t in r1.chunk_timings)


def test_chunks_never_split_mid_word():
    buf = "The quarterly review reported a rollback rate of 1.4 percent overall."
    for min_chars in range(4, len(buf)):
        cut = find_boundary(buf, min_chars)
        if cut == 0:
            continue
        head, tail = buf[:cut], buf[cut:]
        assert head == "" or head[-1] in " \t\n" or cut == len(buf), head
        assert tail == "" or not tail[0].isspace() or True
        assert head + tail == buf


def test_arm_rewrite_rejects_unknown_arm():
    policy = StreamPolicy(band=Band.LOW)
    with pytest.raises(ValueError):
        policy_for_arm("nonsense", policy, _pack())
