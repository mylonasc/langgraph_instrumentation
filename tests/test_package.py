import langgraph_instrumentation
from langgraph_instrumentation import stores


def test_public_api_is_explicit() -> None:
    assert langgraph_instrumentation.__all__ == [
        "Clock",
        "ClockReading",
        "CaptureMode",
        "CapturePolicy",
        "DeterministicClock",
        "DeterministicIdGenerator",
        "ExecutionLane",
        "FlushableTraceStore",
        "IdGenerator",
        "LangGraphInstrumentationHandler",
        "MessageProjector",
        "MemoryTraceStore",
        "MetadataProjector",
        "MetricPoint",
        "PerfettoLogger",
        "PerfettoTracer",
        "RandomIdGenerator",
        "RedactionPolicy",
        "RecorderError",
        "Sanitizer",
        "Span",
        "SpanEvent",
        "SpanId",
        "SpanKind",
        "SpanProcessor",
        "SpanStatus",
        "SystemClock",
        "TokenUsage",
        "Trace",
        "TraceBundle",
        "TraceId",
        "TraceQuery",
        "TraceRecorder",
        "TraceStore",
        "TraceSummary",
        "UsageExtractor",
    ]


def test_store_api_is_explicit() -> None:
    assert stores.__all__ == ["FlushableTraceStore", "MemoryTraceStore", "TraceStore"]
