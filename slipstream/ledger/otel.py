"""OpenTelemetry emission for decision records (§7.9).

Attribute names follow the OpenTelemetry GenAI semantic conventions. Those
conventions are still **Development** status upstream and have already changed
names once (``gen_ai.usage.prompt_tokens`` -> ``gen_ai.usage.input_tokens``), so
the exact set this build emits is pinned in ``ATTRIBUTE_SET`` below and repeated
in the README. Slipstream-specific attributes are namespaced ``slipstream.*``
rather than invented under ``gen_ai.*``.

Exporters: console (opt-in) and a JSONL file exporter, so the demo has something
to show without a collector running.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Sequence

from opentelemetry import trace
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import (
    ConsoleSpanExporter,
    SimpleSpanProcessor,
    SpanExporter,
    SpanExportResult,
)

from .. import settings
from ..contracts import DecisionRecord

# Pinned attribute set. Anything not on this list is not emitted.
ATTRIBUTE_SET = (
    "gen_ai.operation.name",
    "gen_ai.system",
    "gen_ai.request.model",
    "gen_ai.response.model",
    "gen_ai.usage.input_tokens",
    "gen_ai.usage.output_tokens",
    "slipstream.band",
    "slipstream.action",
    "slipstream.arm",
    "slipstream.use_case",
    "slipstream.geography",
    "slipstream.degraded_mode",
    "slipstream.abstained",
    "slipstream.signal_mode",
    "slipstream.entities_masked",
    "slipstream.detectors",
    "slipstream.record_hash",
    "slipstream.ttft_ms",
    "slipstream.total_latency_ms",
    "slipstream.guard_overhead_ms",
    "slipstream.pct_guard_finished_first",
)


class JsonlSpanExporter(SpanExporter):
    """Minimal file exporter. One JSON object per span, appended."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        try:
            with self._lock, self.path.open("a", encoding="utf-8") as fh:
                for s in spans:
                    fh.write(
                        json.dumps(
                            {
                                "name": s.name,
                                "trace_id": f"{s.context.trace_id:032x}",
                                "span_id": f"{s.context.span_id:016x}",
                                "start_ns": s.start_time,
                                "end_ns": s.end_time,
                                "attributes": dict(s.attributes or {}),
                            }
                        )
                        + "\n"
                    )
        except Exception:  # noqa: BLE001 - telemetry must never break the gateway
            return SpanExportResult.FAILURE
        return SpanExportResult.SUCCESS

    def shutdown(self) -> None:
        return None


_provider: TracerProvider | None = None
_tracer = None


def _tracer_instance():
    global _provider, _tracer
    if _tracer is None:
        _provider = TracerProvider()
        _provider.add_span_processor(SimpleSpanProcessor(JsonlSpanExporter(settings.OTEL_FILE)))
        if settings.OTEL_CONSOLE:
            _provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))
        trace.set_tracer_provider(_provider)
        _tracer = trace.get_tracer("slipstream")
    return _tracer


def emit_span(record: DecisionRecord, *, pct_guard_first: float = 0.0) -> None:
    tracer = _tracer_instance()
    with tracer.start_as_current_span("chat " + record.model_used) as span:
        span.set_attribute("gen_ai.operation.name", "chat")
        span.set_attribute("gen_ai.system", "slipstream")
        span.set_attribute("gen_ai.request.model", record.model_used)
        span.set_attribute("gen_ai.response.model", record.model_used)
        span.set_attribute("gen_ai.usage.input_tokens", record.tokens_in)
        span.set_attribute("gen_ai.usage.output_tokens", record.tokens_out)
        span.set_attribute("slipstream.band", record.policy.band.value)
        span.set_attribute("slipstream.action", record.action.value)
        span.set_attribute("slipstream.arm", record.arm)
        span.set_attribute("slipstream.use_case", record.use_case)
        span.set_attribute("slipstream.geography", record.geography)
        span.set_attribute("slipstream.degraded_mode", record.degraded_mode)
        span.set_attribute("slipstream.abstained", record.abstained)
        span.set_attribute("slipstream.signal_mode", record.signal_mode.value)
        span.set_attribute("slipstream.entities_masked", list(record.entities_masked))
        span.set_attribute("slipstream.detectors", list(record.policy.detectors))
        span.set_attribute("slipstream.record_hash", record.record_hash)
        span.set_attribute("slipstream.ttft_ms", record.ttft_ms)
        span.set_attribute("slipstream.total_latency_ms", record.total_latency_ms)
        span.set_attribute("slipstream.guard_overhead_ms", record.guard_overhead_ms)
        span.set_attribute("slipstream.pct_guard_finished_first", pct_guard_first)


def shutdown_telemetry() -> None:
    global _provider, _tracer
    if _provider is not None:
        _provider.shutdown()
    _provider = None
    _tracer = None
