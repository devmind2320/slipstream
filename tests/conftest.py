from __future__ import annotations

import os

# Tests run against measured detector latency, not the modelled §4 targets: the
# latency model exists to make the *eval* comparison meaningful, and paying
# 200 ms per expensive detector in unit tests buys nothing. Must be set before
# slipstream.settings is imported.
os.environ.setdefault("SLIPSTREAM_LATENCY_PROFILE", "measured")
os.environ.setdefault("SLIPSTREAM_TOKEN_MS", "1.0")

import pytest  # noqa: E402

from fixtures_data import PII_PROMPT, PII_RESPONSE  # noqa: E402
from slipstream.gateway.pipeline import SlipstreamGateway  # noqa: E402
from slipstream.gateway.providers.replay import ReplayProvider  # noqa: E402
from slipstream.guards.registry import build_registry  # noqa: E402
from slipstream.policy.loader import PolicyLoader  # noqa: E402


@pytest.fixture
def loader() -> PolicyLoader:
    return PolicyLoader()


@pytest.fixture
def registry():
    return build_registry()


@pytest.fixture
def provider() -> ReplayProvider:
    return ReplayProvider(
        [
            {
                "fixture_id": "pii",
                "prompt": PII_PROMPT,
                "response": PII_RESPONSE,
                "model": "replay-frontier",
            }
        ],
        time_scale=1.0,
    )


@pytest.fixture
def gateway(provider, registry, loader) -> SlipstreamGateway:
    return SlipstreamGateway(
        provider, registry=registry, loader=loader, emit_telemetry=False
    )
