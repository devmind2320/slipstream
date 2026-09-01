"""FastAPI gateway.

    uvicorn slipstream.gateway.app:app --port 8000

Endpoints:
  POST /v1/chat/completions   OpenAI-shaped, SSE when stream=true
  GET  /demo/stream           the A/B demo feed used by the dashboard (§11)
  GET  /demo/scenarios        scenarios the dashboard offers
  GET  /policy/packs          loaded policy packs
  GET  /ledger/recent         recent decision records, hash chain visible
  GET  /ledger/verify         walk the chain, report the first break
  GET  /results               eval results, once `make eval` has run
  GET  /healthz
  GET  /                      dashboard shell; assets under /static

Offline by default: the replay provider is the default and no key is read
anywhere on the demo path (INV-5).
"""

from __future__ import annotations

import asyncio
import json
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .. import settings
from ..contracts import Band
from ..guards.registry import default_registry
from ..ledger.store import Ledger, verify_chain
from ..policy.loader import get_loader
from ..risk.features import ToolScope
from ..routing.cache import SemanticCache
from .pipeline import SlipstreamGateway, SlipstreamRequest
from .providers.replay import ReplayProvider

STATE: dict[str, Any] = {}


def build_provider():
    if settings.PROVIDER == "openai":
        from .providers.openai_compat import OpenAICompatProvider

        return OpenAICompatProvider()
    provider = ReplayProvider.from_dir()
    from ..demo.scenario import DEMO_FIXTURES

    provider.load(DEMO_FIXTURES)
    return provider


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings.ensure_dirs()
    provider = build_provider()
    if hasattr(provider, "probe"):
        await provider.probe()
    ledger = Ledger()
    STATE["provider"] = provider
    STATE["ledger"] = ledger
    STATE["gateway"] = SlipstreamGateway(
        provider,
        registry=default_registry(),
        loader=get_loader(),
        ledger=ledger,
        cache=SemanticCache(),
    )
    yield
    ledger.close()


app = FastAPI(title="Slipstream", version="0.1.0", lifespan=lifespan)

DASHBOARD_DIR = settings.PKG_ROOT / "dashboard"
app.mount(
    "/static",
    StaticFiles(directory=DASHBOARD_DIR / "static"),
    name="static",
)


# ---------------------------------------------------------------------------
class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    model: str = "default"
    messages: list[ChatMessage]
    stream: bool = True
    use_case: str = "internal_knowledge"
    geography: str = "IN"
    retrieval_context: list[str] = Field(default_factory=list)
    conversation_id: str | None = None
    tool_scope: str = "none"
    arm: str = "adaptive"
    fixture_id: str | None = None


def _to_request(body: ChatRequest) -> SlipstreamRequest:
    user = next(
        (m.content for m in reversed(body.messages) if m.role == "user"), ""
    )
    return SlipstreamRequest(
        prompt=user,
        use_case=body.use_case,
        geography=body.geography,
        retrieval_context=body.retrieval_context,
        conversation_id=body.conversation_id,
        tool_scope=ToolScope(body.tool_scope),
        arm=body.arm,
        fixture_id=body.fixture_id,
    )


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


async def _stream_response(req: SlipstreamRequest) -> AsyncIterator[str]:
    gw: SlipstreamGateway = STATE["gateway"]
    try:
        async for ev in gw.run(req):
            if ev.type == "chunk":
                yield _sse(
                    "chunk",
                    {
                        "delta": ev.text,
                        "index": ev.index,
                        "masked_types": ev.masked_types,
                        "meta": ev.meta,
                    },
                )
            elif ev.type in ("blocked", "escalated", "annotation"):
                yield _sse(ev.type, {"text": ev.text, "meta": ev.meta})
            elif ev.type == "done":
                yield _sse("done", ev.meta)
        outcome = gw.last_outcome
        yield _sse("record", json.loads(outcome.record.model_dump_json()))
    except KeyError as exc:
        yield _sse("error", {"message": str(exc)})
    yield "data: [DONE]\n\n"


@app.post("/v1/chat/completions")
async def chat_completions(body: ChatRequest):
    req = _to_request(body)
    if body.stream:
        return StreamingResponse(
            _stream_response(req),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )
    gw: SlipstreamGateway = STATE["gateway"]
    try:
        outcome = await gw.collect(req)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return JSONResponse(
        {
            "id": outcome.record.record_id,
            "object": "chat.completion",
            "model": outcome.record.model_used,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": outcome.result.released_text,
                    },
                    "finish_reason": outcome.record.action.value,
                }
            ],
            "usage": {
                "prompt_tokens": outcome.record.tokens_in,
                "completion_tokens": outcome.record.tokens_out,
            },
            "slipstream": json.loads(outcome.record.model_dump_json()),
        }
    )


# ---------------------------------------------------------------------------
@app.get("/demo/stream")
async def demo_stream(arm: str = "adaptive", scenario: str = "compensation"):
    """Drives one pane of the §11 A/B demo."""
    from ..demo.scenario import get_scenario

    sc = get_scenario(scenario)
    req = SlipstreamRequest(
        prompt=sc["prompt"],
        use_case=sc["use_case"],
        geography=sc.get("geography", "IN"),
        retrieval_context=sc.get("retrieval_context", []),
        arm=arm,
        fixture_id=sc["fixture_id"],
        use_cache=False,
    )
    return StreamingResponse(
        _stream_response(req),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/demo/scenarios")
async def demo_scenarios():
    from ..demo.scenario import SCENARIOS

    return {
        k: {"title": v["title"], "prompt": v["prompt"], "use_case": v["use_case"]}
        for k, v in SCENARIOS.items()
    }


@app.get("/policy/packs")
async def policy_packs():
    loader = get_loader()
    return {
        name: json.loads(loader.load(name).model_dump_json())
        for name in loader.available()
    }


@app.get("/ledger/recent")
async def ledger_recent(limit: int = 20):
    ledger: Ledger = STATE["ledger"]
    return {"count": len(ledger), "records": ledger.recent(limit)}


@app.get("/ledger/verify")
async def ledger_verify():
    ledger: Ledger = STATE["ledger"]
    brk = verify_chain(ledger)
    if brk is None:
        return {"ok": True, "records": len(ledger), "message": "hash chain intact"}
    return {
        "ok": False,
        "records": len(ledger),
        "first_break": {
            "seq": brk.seq,
            "record_id": brk.record_id,
            "kind": brk.kind,
            "expected": brk.expected,
            "found": brk.found,
        },
    }


@app.get("/results")
async def results():
    path = settings.RESULTS_DIR / "results.json"
    if not path.exists():
        raise HTTPException(404, "no results yet; run `make eval`")
    return JSONResponse(json.loads(path.read_text(encoding="utf-8")))


@app.get("/healthz")
async def healthz():
    return {
        "ok": True,
        "provider": STATE["provider"].name,
        "backends": settings.backend_summary(),
        "detectors": default_registry().names,
        "registry_fingerprint": default_registry().fingerprint(),
        "packs": get_loader().available(),
    }


@app.get("/", response_class=HTMLResponse)
async def dashboard():
    path = DASHBOARD_DIR / "index.html"
    # no-store: the shell is tiny and a stale one during a live demo is worse
    # than re-reading it. Assets under /static carry their own cache headers.
    return HTMLResponse(
        path.read_text(encoding="utf-8"),
        headers={"Cache-Control": "no-store"},
    )
