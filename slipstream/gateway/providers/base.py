"""Provider protocol.

Two implementations exist (replay, openai_compat) so this is a real seam, not an
abstraction built for imaginary future providers.

The capability probe is the important part. Per §7.4 the system must not assume
logprobs are available, because most enterprise API deployments do not return
them. Providers declare what they can do; the risk estimator picks its signal set
accordingly and records which one it used.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import AsyncIterator, Protocol, runtime_checkable

from ...contracts import ProviderCapabilities


@dataclass
class GenerationRequest:
    prompt: str
    system: str | None = None
    model: str = "default"
    max_tokens: int = 512
    temperature: float = 0.0
    fixture_id: str | None = None  # replay provider addressing
    metadata: dict = field(default_factory=dict)


@dataclass
class TokenEvent:
    text: str
    logprob: float | None = None
    index: int = 0


@runtime_checkable
class Provider(Protocol):
    name: str

    def capabilities(self) -> ProviderCapabilities: ...

    def stream(self, req: GenerationRequest) -> AsyncIterator[TokenEvent]:
        """Yield tokens as they are generated."""
        ...

    async def sample(self, req: GenerationRequest, n: int) -> list[str]:
        """n independent completions. Only called when supports_multi_sample."""
        ...
