"""OpenAI-compatible streaming provider. Optional — never used by make demo/eval.

Enabled with SLIPSTREAM_PROVIDER=openai plus SLIPSTREAM_OPENAI_BASE_URL and
SLIPSTREAM_OPENAI_API_KEY. Works against any /v1/chat/completions endpoint that
speaks SSE (vLLM, Ollama's compat shim, Together, OpenAI itself).

The capability probe here is deliberately conservative: it asks for one token
with logprobs, and if the field comes back missing or the request errors, it
records supports_logprobs=False and the proxy signal path is used instead.
"""

from __future__ import annotations

import json
from typing import AsyncIterator

from ...contracts import ProviderCapabilities
from ... import settings
from .base import GenerationRequest, TokenEvent


class OpenAICompatProvider:
    name = "openai_compat"

    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        model: str = "gpt-4o-mini",
    ) -> None:
        self.base_url = (base_url or settings.OPENAI_BASE_URL).rstrip("/")
        self.api_key = api_key or settings.OPENAI_API_KEY
        self.model = model
        self._caps: ProviderCapabilities | None = None
        if not self.base_url:
            raise RuntimeError("SLIPSTREAM_OPENAI_BASE_URL is required for this provider")

    def _client(self):
        import httpx  # imported lazily so the offline path never needs it

        return httpx.AsyncClient(
            base_url=self.base_url,
            headers={"Authorization": f"Bearer {self.api_key}"},
            timeout=60.0,
        )

    async def probe(self) -> ProviderCapabilities:
        supports_logprobs = False
        try:
            async with self._client() as c:
                r = await c.post(
                    "/v1/chat/completions",
                    json={
                        "model": self.model,
                        "messages": [{"role": "user", "content": "hi"}],
                        "max_tokens": 1,
                        "logprobs": True,
                        "top_logprobs": 1,
                    },
                )
                if r.status_code == 200:
                    body = r.json()
                    supports_logprobs = bool(
                        body.get("choices", [{}])[0].get("logprobs")
                    )
        except Exception:  # noqa: BLE001 - probe failure means "assume not supported"
            supports_logprobs = False
        self._caps = ProviderCapabilities(
            supports_logprobs=supports_logprobs,
            supports_multi_sample=True,
            supports_system_prompt=True,
            name=self.name,
        )
        return self._caps

    def capabilities(self) -> ProviderCapabilities:
        return self._caps or ProviderCapabilities(name=self.name)

    async def stream(self, req: GenerationRequest) -> AsyncIterator[TokenEvent]:
        caps = self.capabilities()
        messages = []
        if req.system and caps.supports_system_prompt:
            messages.append({"role": "system", "content": req.system})
        messages.append({"role": "user", "content": req.prompt})
        payload = {
            "model": req.model if req.model != "default" else self.model,
            "messages": messages,
            "max_tokens": req.max_tokens,
            "temperature": req.temperature,
            "stream": True,
        }
        if caps.supports_logprobs:
            payload["logprobs"] = True

        idx = 0
        async with self._client() as c:
            async with c.stream("POST", "/v1/chat/completions", json=payload) as r:
                r.raise_for_status()
                async for line in r.aiter_lines():
                    if not line.startswith("data: "):
                        continue
                    body = line[6:].strip()
                    if body == "[DONE]":
                        break
                    try:
                        obj = json.loads(body)
                    except json.JSONDecodeError:
                        continue
                    choice = (obj.get("choices") or [{}])[0]
                    delta = (choice.get("delta") or {}).get("content")
                    if not delta:
                        continue
                    lp = None
                    lpobj = choice.get("logprobs") or {}
                    content = lpobj.get("content") or []
                    if content and isinstance(content[0], dict):
                        lp = content[0].get("logprob")
                    yield TokenEvent(text=delta, logprob=lp, index=idx)
                    idx += 1

    async def sample(self, req: GenerationRequest, n: int) -> list[str]:
        out: list[str] = []
        async with self._client() as c:
            for _ in range(n):
                r = await c.post(
                    "/v1/chat/completions",
                    json={
                        "model": req.model if req.model != "default" else self.model,
                        "messages": [{"role": "user", "content": req.prompt}],
                        "max_tokens": req.max_tokens,
                        "temperature": 0.8,
                    },
                )
                r.raise_for_status()
                out.append(r.json()["choices"][0]["message"]["content"])
        return out
