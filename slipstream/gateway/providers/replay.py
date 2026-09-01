"""Deterministic fixture playback. This is the default provider (INV-5).

A fixture holds the full response text plus a per-token delay schedule derived
from a seeded PRNG keyed on the fixture id. Two runs on the same machine emit
the same tokens in the same order with the same *scheduled* pacing, so the
scheduler's concurrency behaviour is reproducible without a network or a key.

Wall-clock latency still varies machine to machine — asyncio.sleep has jitter and
the detectors are real CPU work. The eval harness therefore separates its
deterministic metric block from its timing block; only the former is asserted
byte-identical across runs.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import random
import re
from pathlib import Path
from typing import AsyncIterator, Iterable

from ...contracts import ProviderCapabilities
from ... import settings
from .base import GenerationRequest, TokenEvent

_SPLIT = re.compile(r"\S+\s*")

# Scales the simulated inter-token delay. 1.0 ~= 5 ms/token, a realistic
# mid-size hosted model. The eval leaves this at 1.0; tests turn it down.
TIME_SCALE = float(os.environ.get("SLIPSTREAM_TIME_SCALE", "1.0"))
BASE_TOKEN_MS = float(os.environ.get("SLIPSTREAM_TOKEN_MS", "5.0"))


def tokenise(text: str) -> list[str]:
    """Whitespace-preserving split, so ''.join(tokens) == text exactly."""
    return _SPLIT.findall(text) or ([text] if text else [])


class Fixture:
    __slots__ = (
        "fixture_id", "prompt", "response", "samples", "model", "logprobs", "constrained",
    )

    def __init__(self, d: dict) -> None:
        self.fixture_id: str = d["fixture_id"]
        self.prompt: str = d.get("prompt", "")
        self.response: str = d.get("response", "")
        self.samples: list[str] = list(d.get("samples", []))
        self.model: str = d.get("model", "replay-sim")
        self.logprobs: list[float] | None = d.get("logprobs")
        # What the model produces on the one permitted retry under a constraint
        # prompt. Absent means the retry changes nothing, which is itself a
        # realistic outcome and is recorded as such.
        self.constrained: str | None = d.get("constrained")


def fixture_key(prompt: str) -> str:
    return hashlib.sha256(prompt.strip().encode("utf-8")).hexdigest()[:16]


class ReplayProvider:
    """Fixture-backed provider.

    ``supports_logprobs`` defaults to False on purpose: the honest default for an
    enterprise API deployment is that logprobs are unavailable (§7.4). Flip it on
    to exercise the entropy-based signal path in the eval.
    """

    name = "replay"

    def __init__(
        self,
        fixtures: Iterable[dict] | None = None,
        *,
        supports_logprobs: bool = False,
        supports_multi_sample: bool = True,
        time_scale: float | None = None,
    ) -> None:
        self._by_id: dict[str, Fixture] = {}
        self._by_prompt: dict[str, Fixture] = {}
        self._caps = ProviderCapabilities(
            supports_logprobs=supports_logprobs,
            supports_multi_sample=supports_multi_sample,
            supports_system_prompt=True,
            name=self.name,
        )
        self.time_scale = TIME_SCALE if time_scale is None else time_scale
        if fixtures:
            self.load(fixtures)

    # -- loading ---------------------------------------------------------
    def load(self, fixtures: Iterable[dict]) -> None:
        for d in fixtures:
            f = Fixture(d)
            self._by_id[f.fixture_id] = f
            if f.prompt:
                self._by_prompt[fixture_key(f.prompt)] = f

    @classmethod
    def from_dir(cls, path: Path | None = None, **kw) -> "ReplayProvider":
        path = path or settings.FIXTURES_DIR
        items: list[dict] = []
        if path.exists():
            for p in sorted(path.glob("*.jsonl")):
                with p.open("r", encoding="utf-8") as fh:
                    items.extend(json.loads(line) for line in fh if line.strip())
            for p in sorted(path.glob("*.json")):
                data = json.loads(p.read_text(encoding="utf-8"))
                items.extend(data if isinstance(data, list) else [data])
        return cls(items, **kw)

    def add(self, fixture_id: str, prompt: str, response: str, **kw) -> None:
        self.load([{"fixture_id": fixture_id, "prompt": prompt, "response": response, **kw}])

    # -- protocol --------------------------------------------------------
    def capabilities(self) -> ProviderCapabilities:
        return self._caps

    def _resolve(self, req: GenerationRequest) -> Fixture:
        if req.fixture_id and req.fixture_id in self._by_id:
            return self._by_id[req.fixture_id]
        f = self._by_prompt.get(fixture_key(req.prompt))
        if f is not None:
            return f
        raise KeyError(
            f"no fixture for id={req.fixture_id!r} / prompt hash "
            f"{fixture_key(req.prompt)}. Run `make data` to regenerate fixtures."
        )

    def _delays(self, f: Fixture, n: int) -> list[float]:
        rng = random.Random(int(fixture_key(f.fixture_id or f.prompt), 16))
        # Lognormal-ish jitter around BASE_TOKEN_MS; first token costs more,
        # which is what a real prefill looks like.
        out = [max(0.0, rng.gauss(BASE_TOKEN_MS, BASE_TOKEN_MS * 0.35)) for _ in range(n)]
        if out:
            out[0] += BASE_TOKEN_MS * 6
        return [d * self.time_scale / 1000.0 for d in out]

    async def stream(self, req: GenerationRequest) -> AsyncIterator[TokenEvent]:
        f = self._resolve(req)
        body = f.response
        if req.metadata.get("variant") == "constrained" and f.constrained:
            body = f.constrained
        toks = tokenise(body)
        delays = self._delays(f, len(toks))
        lps = f.logprobs if (f.logprobs and self._caps.supports_logprobs) else None
        for i, (tok, delay) in enumerate(zip(toks, delays)):
            if delay > 0:
                await asyncio.sleep(delay)
            yield TokenEvent(
                text=tok,
                logprob=(lps[i] if lps and i < len(lps) else None),
                index=i,
            )

    async def sample(self, req: GenerationRequest, n: int) -> list[str]:
        f = self._resolve(req)
        if not self._caps.supports_multi_sample:
            return []
        pool = [f.response] + f.samples
        # Deterministic: cycle the available variants rather than draw randomly.
        return [pool[i % len(pool)] for i in range(n)]

    def model_for(self, req: GenerationRequest) -> str:
        try:
            return self._resolve(req).model
        except KeyError:
            return "replay-sim"
