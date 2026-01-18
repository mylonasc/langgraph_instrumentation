import json
import time
import threading
import os
import logging
import asyncio
import zlib
from dataclasses import dataclass
from typing import Dict, Any, Optional, List, Set
from uuid import UUID
from collections import OrderedDict

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult
from langchain_core.messages import BaseMessage


# =============================================================================
# TRACER
# =============================================================================

class PerfettoTracer:
    """
    Drop-in compatible tracer with a few robustness upgrades:
      - Optional max_events cap to avoid unbounded memory growth
      - Optional pretty JSON output toggle
      - Adds process name metadata (optional; safe no-op if unused)
    """

    def __init__(self, *, max_events: int = 200_000, process_name: Optional[str] = None):
        self.trace_events: List[Dict[str, Any]] = []
        self.pid = os.getpid()
        self._lock = threading.Lock()
        self._max_events = int(max_events) if max_events and max_events > 0 else 0

        if process_name:
            self.set_process_name(process_name)

    @staticmethod
    def _now_us() -> int:
        return time.perf_counter_ns() // 1000

    def _append_event(self, ev: Dict[str, Any]) -> None:
        with self._lock:
            self.trace_events.append(ev)
            if self._max_events and len(self.trace_events) > self._max_events:
                # Keep the most recent events (cheap + predictable).
                # If you prefer a ring buffer, switch trace_events to a deque.
                overflow = len(self.trace_events) - self._max_events
                if overflow > 0:
                    del self.trace_events[:overflow]

    def add_event(
        self,
        name: str,
        phase: str,
        category: str,
        ts_us: Optional[int] = None,
        tid: Optional[int] = None,
        args: Optional[Dict[str, Any]] = None,
        **extra_fields: Any,
    ):
        if ts_us is None:
            ts_us = self._now_us()
        if tid is None:
            tid = threading.get_ident()

        ev = {
            "name": name,
            "cat": category,
            "ph": phase,
            "ts": int(ts_us),
            "pid": int(self.pid),
            "tid": int(tid),
            "args": args or {},
        }
        ev.update(extra_fields)
        self._append_event(ev)

    def add_complete_event(
        self,
        name: str,
        category: str,
        start_ts_us: int,
        dur_us: int,
        tid: int,
        args: Optional[Dict[str, Any]] = None,
    ):
        dur_us = max(1, int(dur_us))
        ev = {
            "name": name,
            "cat": category,
            "ph": "X",
            "ts": int(start_ts_us),
            "dur": dur_us,
            "pid": int(self.pid),
            "tid": int(tid),
            "args": args or {},
        }
        self._append_event(ev)

    def add_counter(
        self,
        name: str,
        values: Dict[str, Any],
        ts_us: Optional[int] = None,
        tid: Optional[int] = None,
    ):
        self.add_event(name=name, phase="C", category="metrics", ts_us=ts_us, tid=tid, args=values)

    def set_thread_name(self, tid: int, name: str):
        # Trace Event Format metadata
        self.add_event(
            name="thread_name",
            phase="M",
            category="__metadata",
            tid=tid,
            args={"name": name},
        )

    def set_process_name(self, name: str):
        self.add_event(
            name="process_name",
            phase="M",
            category="__metadata",
            tid=0,
            args={"name": name},
        )

    def set_thread_sort_index(self, tid: int, sort_index: int):
        self.add_event(
            name="thread_sort_index",
            phase="M",
            category="__metadata",
            tid=tid,
            args={"sort_index": int(sort_index)},
        )

    # Flow events: start/end of a flow arrow across threads/tracks
    def add_flow_start(self, flow_id: int, ts_us: int, tid: int, args: Dict[str, Any]):
        self.add_event(
            name="flow",
            phase="s",
            category="flow",
            ts_us=ts_us,
            tid=tid,
            args=args,
            id=int(flow_id),
            bp="e",
        )

    def add_flow_end(self, flow_id: int, ts_us: int, tid: int, args: Dict[str, Any]):
        self.add_event(
            name="flow",
            phase="f",
            category="flow",
            ts_us=ts_us,
            tid=tid,
            args=args,
            id=int(flow_id),
            bp="e",
        )

    def save(self, filename: str, *, pretty: bool = True):
        with self._lock:
            snapshot = list(self.trace_events)
        os.makedirs(os.path.dirname(filename) or ".", exist_ok=True)
        with open(filename, "w") as f:
            if pretty:
                json.dump({"traceEvents": snapshot}, f, indent=2, default=str)
            else:
                json.dump({"traceEvents": snapshot}, f, default=str)


# =============================================================================
# LOGGER
# =============================================================================

class PerfettoLogger:
    """
    Drop-in compatible logger with safer defaults:
      - doesn't duplicate handlers
      - you can pass an existing logger name
    """
    def __init__(self, logger_name: str = "instrumentation"):
        self.logger = logging.getLogger(logger_name)

        # Only add a default handler if the user hasn't configured one.
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter("%(asctime)s | %(message)s", datefmt="%H:%M:%S")
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)

        # Avoid double-logging unless the user wants propagation.
        self.logger.propagate = False

    def log_step(self, phase: str, category: str, name: str, details: str = ""):
        icon_map = {"START": "🟢", "END": "🔴", "ERROR": "❌"}
        icon = icon_map.get(phase, "⚪")
        msg = f"{icon} [{category.upper()}] {name}"
        if details:
            msg += f" | {details}"
        self.logger.info(msg)

    def log_state(self, node_name: str, state_summary: str):
        self.logger.info(f"   📝 STATE [{node_name}]: {state_summary}")


# =============================================================================
# HANDLER
# =============================================================================

@dataclass
class RunInfo:
    name: str
    category: str
    start_ts_us: int
    tid: int
    parent_run_id: Optional[str]


class LangGraphInstrumentationHandler(BaseCallbackHandler):
    """
    Drop-in replacement handler with refactors:
      - Cycle-safe, bounded sanitizer + redaction support
      - Optional caps for trace growth (via tracer) and message-id dedupe set
      - Cleaner flow fix: flow spans from parent's start_ts -> child's start_ts
      - Async-friendly: can optionally use virtual "task tids" to separate concurrent async runs
      - Keeps your virtual tool tid behavior (tool_call_id -> deterministic tid)
    """

    _TOOL_TID_OFFSET = 10_000_000
    _TASK_TID_OFFSET = 20_000_000

    def __init__(
        self,
        tracer: PerfettoTracer,
        logger: PerfettoLogger,
        *,
        # Sanitization controls (safe defaults)
        sanitize_max_depth: int = 6,
        sanitize_max_str: int = 2_000,
        sanitize_max_items: int = 50,
        redact_keys: Optional[Set[str]] = None,
        # Dedup controls
        max_seen_message_ids: int = 50_000,
        # Async lane separation
        use_async_task_tids: bool = True,
    ):
        super().__init__()
        self.tracer = tracer
        self.console = logger

        self.run_map: Dict[str, RunInfo] = {}
        self._run_lock = threading.Lock()

        self._named_tids: Set[int] = set()
        self._named_tids_lock = threading.Lock()

        self._flow_lock = threading.Lock()
        self._flow_counter = 0  # fixed: first id will be 1

        self.token_counts = {"total": 0, "input": 0, "output": 0}

        # LRU-like tracking to avoid repeating large message blobs forever
        self._seen_message_ids = OrderedDict()  # msg_key -> None
        self._msg_lock = threading.Lock()
        self._max_seen_message_ids = int(max_seen_message_ids) if max_seen_message_ids and max_seen_message_ids > 0 else 0

        # Sanitization policy
        self._sanitize_max_depth = int(sanitize_max_depth)
        self._sanitize_max_str = int(sanitize_max_str)
        self._sanitize_max_items = int(sanitize_max_items)
        self._redact_keys = {k.lower() for k in (redact_keys or {"api_key", "authorization", "token", "password", "secret"})}

        self._use_async_task_tids = bool(use_async_task_tids)

    @staticmethod
    def _now_us() -> int:
        return time.perf_counter_ns() // 1000

    def _new_flow_id(self) -> int:
        with self._flow_lock:
            self._flow_counter += 1
            return self._flow_counter

    def _ensure_thread_named(self, tid: int, label: Optional[str] = None):
        with self._named_tids_lock:
            if tid in self._named_tids:
                return
            self._named_tids.add(tid)
        self.tracer.set_thread_name(tid, label or f"py-thread:{tid}")

    def _tool_virtual_tid(self, tool_call_id: str) -> int:
        h = zlib.crc32(tool_call_id.encode("utf-8")) & 0xFFFFFFFF
        return self._TOOL_TID_OFFSET + int(h)

    def _task_virtual_tid(self) -> Optional[int]:
        if not self._use_async_task_tids:
            return None
        try:
            task = asyncio.current_task()
        except RuntimeError:
            task = None
        if not task:
            return None
        # Stable-ish per task during lifetime; sufficient for lane separation.
        return self._TASK_TID_OFFSET + (id(task) & 0xFFFFFFFF)

    def _extract_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        keep_keys = {"thread_id", "langgraph_node", "checkpoint_id", "graph_name"}
        return {k: metadata[k] for k in keep_keys if k in metadata}

    # ---------------------------
    # Sanitization (bounded + cycle-safe + redaction)
    # ---------------------------

    def _sanitize(self, obj: Any, *, _depth: int = 0, _seen: Optional[Set[int]] = None) -> Any:
        if _seen is None:
            _seen = set()
        oid = id(obj)
        if oid in _seen:
            return "<cycle>"
        if _depth >= self._sanitize_max_depth:
            return "<max_depth>"

        # primitives
        if obj is None or isinstance(obj, (bool, int, float)):
            return obj
        if isinstance(obj, str):
            return obj if len(obj) <= self._sanitize_max_str else (obj[: self._sanitize_max_str] + "…<trunc>")
        if isinstance(obj, bytes):
            return f"<bytes:{len(obj)}>"

        _seen.add(oid)

        # dict-like
        if isinstance(obj, dict):
            out: Dict[str, Any] = {}
            items = list(obj.items())
            for k, v in items[: self._sanitize_max_items]:
                ks = str(k)
                if ks.lower() in self._redact_keys:
                    out[ks] = "<redacted>"
                else:
                    out[ks] = self._sanitize(v, _depth=_depth + 1, _seen=_seen)
            if len(items) > self._sanitize_max_items:
                out["…"] = f"<{len(items) - self._sanitize_max_items} more keys>"
            return out

        # list/tuple/set
        if isinstance(obj, (list, tuple, set)):
            seq = list(obj)
            out = [self._sanitize(x, _depth=_depth + 1, _seen=_seen) for x in seq[: self._sanitize_max_items]]
            if len(seq) > self._sanitize_max_items:
                out.append(f"…<{len(seq) - self._sanitize_max_items} more items>")
            return out

        # BaseMessage gets special handling to avoid enormous nested structures
        if isinstance(obj, BaseMessage):
            return self._process_single_message(obj)

        # fallback for other objects
        return f"<{obj.__class__.__name__}:{str(obj)[: self._sanitize_max_str]}>"

    def _mark_message_seen(self, msg_key: str) -> bool:
        """
        Returns True if already seen, False if newly recorded.
        LRU behavior if max cap is set.
        """
        with self._msg_lock:
            if msg_key in self._seen_message_ids:
                # refresh LRU
                self._seen_message_ids.move_to_end(msg_key, last=True)
                return True
            self._seen_message_ids[msg_key] = None
            if self._max_seen_message_ids and len(self._seen_message_ids) > self._max_seen_message_ids:
                self._seen_message_ids.popitem(last=False)
            return False

    def _process_single_message(self, message: Any) -> Any:
        if not isinstance(message, BaseMessage):
            return self._sanitize(message)

        msg_id = getattr(message, "id", None)
        if msg_id is None:
            msg_key = None
        elif isinstance(msg_id, (str, int)):
            msg_key = str(msg_id)
        else:
            msg_key = json.dumps(msg_id, default=str, sort_keys=True)

        fields_to_keep = ["type", "name", "content", "id", "tool_call_id", "additional_kwargs"]
        msg_data: Dict[str, Any] = {}
        for f in fields_to_keep:
            if hasattr(message, f):
                msg_data[f] = getattr(message, f)

        # sanitize potentially huge / sensitive fields
        if "content" in msg_data:
            msg_data["content"] = self._sanitize(msg_data["content"])
        if "additional_kwargs" in msg_data:
            msg_data["additional_kwargs"] = self._sanitize(msg_data["additional_kwargs"])

        if msg_key is not None:
            if self._mark_message_seen(msg_key):
                ref = {"id": msg_id}
                if hasattr(message, "type"):
                    ref["type"] = getattr(message, "type")
                return ref

        return msg_data

    # ---------------------------
    # Run tracking + flows
    # ---------------------------

    def _choose_tid(self, tid_override: Optional[int] = None) -> int:
        if tid_override is not None:
            return tid_override

        # If we're in an async Task, optionally create a virtual tid lane for concurrency.
        task_tid = self._task_virtual_tid()
        if task_tid is not None:
            return task_tid

        return threading.get_ident()

    def _start_run(
        self,
        run_id: UUID,
        name: str,
        category: str,
        args: Optional[Dict[str, Any]],
        parent_run_id: Optional[UUID],
        tid_override: Optional[int] = None,
    ):
        rid = str(run_id)
        start_ts = self._now_us()
        tid = self._choose_tid(tid_override)

        self._ensure_thread_named(tid)

        parent_key = str(parent_run_id) if parent_run_id is not None else None

        # FLOW FIX (cleanest option):
        # Emit a flow that spans from parent.start_ts_us -> child.start_ts_us.
        # This avoids "zero-length" flows while requiring no extra bookkeeping.
        if parent_key is not None:
            with self._run_lock:
                parent_info = self.run_map.get(parent_key)
            if parent_info is not None:
                flow_id = self._new_flow_id()
                flow_args = {
                    "parent": parent_info.name,
                    "child": name,
                    "parent_run_id": parent_key,
                    "child_run_id": rid,
                }
                self.tracer.add_flow_start(flow_id, parent_info.start_ts_us, parent_info.tid, flow_args)
                self.tracer.add_flow_end(flow_id, start_ts, tid, flow_args)

        with self._run_lock:
            self.run_map[rid] = RunInfo(
                name=name,
                category=category,
                start_ts_us=start_ts,
                tid=tid,
                parent_run_id=parent_key,
            )

        self.console.log_step("START", category, name)

        if args:
            marker = dict(args)
            marker["parent_run_id"] = parent_key
            marker["callback_tid"] = threading.get_ident()
            self.tracer.add_event(
                name=name,
                phase="i",
                category=category,
                ts_us=start_ts,
                tid=tid,
                args={"start": marker},
                s="t",
            )

    def _end_run(self, run_id: UUID, end_args: Optional[Dict[str, Any]] = None, error: Optional[str] = None):
        rid = str(run_id)
        with self._run_lock:
            info = self.run_map.pop(rid, None)

        if info is None:
            # Still emit something visible, but keep it cheap.
            ts = self._now_us()
            args = end_args or {}
            if error:
                args = dict(args)
                args["error"] = error
            self.tracer.add_complete_event("Unknown", "unknown", ts, 1, self._choose_tid(), args=args)
            return

        end_ts = self._now_us()
        dur = max(1, end_ts - info.start_ts_us)

        args: Dict[str, Any] = {}
        if end_args:
            args.update(end_args)
        if error:
            args["error"] = error

        args["parent_run_id"] = info.parent_run_id
        args["start_tid"] = info.tid
        args["end_callback_tid"] = threading.get_ident()

        # End on the stored tid so virtual tool/task lanes stay consistent even if callbacks fire elsewhere.
        self.tracer.add_complete_event(info.name, info.category, info.start_ts_us, dur, info.tid, args=args)

        if error:
            self.console.log_step("ERROR", info.category, info.name, error)
        else:
            self.console.log_step("END", info.category, info.name)

    # ---------------------------
    # LLM usage extraction
    # ---------------------------

    def _extract_usage(self, response: LLMResult) -> Dict[str, int]:
        llm_out = getattr(response, "llm_output", None) or {}
        tok = llm_out.get("token_usage") or llm_out.get("usage") or {}

        if isinstance(tok, dict) and ("total_tokens" in tok or "prompt_tokens" in tok or "completion_tokens" in tok):
            input_t = tok.get("prompt_tokens", tok.get("input_tokens", 0)) or 0
            output_t = tok.get("completion_tokens", tok.get("output_tokens", 0)) or 0
            total_t = tok.get("total_tokens", (input_t + output_t)) or 0
            return {"input_tokens": int(input_t), "output_tokens": int(output_t), "total_tokens": int(total_t)}

        try:
            gen0 = response.generations[0][0]
            msg = getattr(gen0, "message", None)
            usage2 = getattr(msg, "usage_metadata", None) or {}
            if isinstance(usage2, dict) and ("total_tokens" in usage2 or "input_tokens" in usage2 or "output_tokens" in usage2):
                input_t = usage2.get("input_tokens", 0) or 0
                output_t = usage2.get("output_tokens", 0) or 0
                total_t = usage2.get("total_tokens", (input_t + output_t)) or 0
                return {"input_tokens": int(input_t), "output_tokens": int(output_t), "total_tokens": int(total_t)}
        except Exception:
            pass

        return {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}

    # -------------------------------------------------------------------------
    # CHAIN / NODE (sync)
    # -------------------------------------------------------------------------
    def on_chain_start(
        self,
        serialized: Dict[str, Any],
        inputs: Dict[str, Any],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs,
    ):
        metadata = kwargs.get("metadata", {}) or {}
        name = kwargs.get("name") or "Chain"
        category = "chain"

        try:
            node_name = metadata.get("langgraph_node")
            if node_name:
                name = f"Node: {node_name}"
                category = "graph_node"
                sanitized_input = self._sanitize(inputs)
                self.console.log_state(node_name, f"Input -> {str(sanitized_input)[:100]}...")
            elif "graph_name" in metadata:
                name = f"Graph: {metadata['graph_name']}"
                category = "graph_root"
                sanitized_input = self._sanitize(inputs)
            else:
                sanitized_input = self._sanitize(inputs)
        except Exception as e:
            sanitized_input = {"instrumentation_error": str(e)}
            self.console.log_step("ERROR", "instrumentation", "on_chain_start", str(e))

        trace_args = {**self._extract_metadata(metadata), "inputs": sanitized_input}
        if parent_run_id is not None:
            trace_args["parent_run_id"] = str(parent_run_id)

        self._start_run(run_id, name, category, trace_args, parent_run_id)

    def on_chain_end(self, outputs: Dict[str, Any], *, run_id: UUID, **kwargs):
        sanitized_output = self._sanitize(outputs)
        self._end_run(run_id, end_args={"outputs": sanitized_output})

    def on_chain_error(self, error: BaseException, *, run_id: UUID, **kwargs):
        self._end_run(run_id, error=str(error))

    # -------------------------------------------------------------------------
    # CHAIN / NODE (async) — delegates to sync logic
    # -------------------------------------------------------------------------
    async def aon_chain_start(
        self,
        serialized: Dict[str, Any],
        inputs: Dict[str, Any],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs,
    ):
        self.on_chain_start(serialized, inputs, run_id=run_id, parent_run_id=parent_run_id, **kwargs)

    async def aon_chain_end(self, outputs: Dict[str, Any], *, run_id: UUID, **kwargs):
        self.on_chain_end(outputs, run_id=run_id, **kwargs)

    async def aon_chain_error(self, error: BaseException, *, run_id: UUID, **kwargs):
        self.on_chain_error(error, run_id=run_id, **kwargs)

    # -------------------------------------------------------------------------
    # LLM (sync)
    # -------------------------------------------------------------------------
    def on_chat_model_start(
        self,
        serialized: Dict[str, Any],
        messages: List[List[BaseMessage]],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs,
    ):
        model = (kwargs.get("invocation_params", {}) or {}).get("model", "ChatModel")
        self._start_run(run_id, f"LLM: {model}", "llm", {"model": model}, parent_run_id)

    def on_llm_start(
        self,
        serialized: Dict[str, Any],
        prompts: List[str],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs,
    ):
        model = (kwargs.get("invocation_params", {}) or {}).get("model", "LLM")
        self._start_run(run_id, f"LLM: {model}", "llm", {"model": model}, parent_run_id)

    def on_llm_end(self, response: LLMResult, *, run_id: UUID, **kwargs):
        usage = self._extract_usage(response)
        total_t = usage["total_tokens"]
        if total_t > 0:
            self.token_counts["total"] += total_t
            self.token_counts["input"] += usage["input_tokens"]
            self.token_counts["output"] += usage["output_tokens"]

            with self._run_lock:
                info = self.run_map.get(str(run_id))
            tid_for_metrics = info.tid if info else None
            self.tracer.add_counter("Token Usage", dict(self.token_counts), ts_us=self._now_us(), tid=tid_for_metrics)

        self._end_run(run_id, end_args={"tokens": usage})

    def on_llm_error(self, error: BaseException, *, run_id: UUID, **kwargs):
        self._end_run(run_id, error=str(error))

    # -------------------------------------------------------------------------
    # LLM (async) — delegates to sync logic
    # -------------------------------------------------------------------------
    async def aon_chat_model_start(
        self,
        serialized: Dict[str, Any],
        messages: List[List[BaseMessage]],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs,
    ):
        self.on_chat_model_start(serialized, messages, run_id=run_id, parent_run_id=parent_run_id, **kwargs)

    async def aon_llm_start(
        self,
        serialized: Dict[str, Any],
        prompts: List[str],
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs,
    ):
        self.on_llm_start(serialized, prompts, run_id=run_id, parent_run_id=parent_run_id, **kwargs)

    async def aon_llm_end(self, response: LLMResult, *, run_id: UUID, **kwargs):
        self.on_llm_end(response, run_id=run_id, **kwargs)

    async def aon_llm_error(self, error: BaseException, *, run_id: UUID, **kwargs):
        self.on_llm_error(error, run_id=run_id, **kwargs)

    # -------------------------------------------------------------------------
    # TOOL (sync) — tool_call_id -> virtual tid track
    # -------------------------------------------------------------------------
    def on_tool_start(
        self,
        serialized: Dict[str, Any],
        input_str: str,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs,
    ):
        if parent_run_id is None and kwargs.get("parent_run_id") is not None:
            parent_run_id = kwargs["parent_run_id"]

        tool_name = serialized.get("name") or "Unknown"
        tool_call_id = kwargs.get("tool_call_id")

        virtual_tid: Optional[int] = None
        if tool_call_id:
            tool_call_id = str(tool_call_id)
            virtual_tid = self._tool_virtual_tid(tool_call_id)
            self._ensure_thread_named(virtual_tid, f"tool:{tool_name}:{tool_call_id}")

        self._start_run(
            run_id,
            f"Tool: {tool_name}",
            "tool",
            {"input": self._sanitize(input_str), "tool_call_id": tool_call_id},
            parent_run_id,
            tid_override=virtual_tid,
        )

    def on_tool_end(self, output: Any, *, run_id: UUID, **kwargs):
        safe_output = self._sanitize(output)
        self._end_run(run_id, end_args={"output": safe_output})

    def on_tool_error(self, error: BaseException, *, run_id: UUID, **kwargs):
        self._end_run(run_id, error=str(error))

    # -------------------------------------------------------------------------
    # TOOL (async) — delegates to sync logic
    # -------------------------------------------------------------------------
    async def aon_tool_start(
        self,
        serialized: Dict[str, Any],
        input_str: str,
        *,
        run_id: UUID,
        parent_run_id: Optional[UUID] = None,
        **kwargs,
    ):
        self.on_tool_start(serialized, input_str, run_id=run_id, parent_run_id=parent_run_id, **kwargs)

    async def aon_tool_end(self, output: Any, *, run_id: UUID, **kwargs):
        self.on_tool_end(output, run_id=run_id, **kwargs)

    async def aon_tool_error(self, error: BaseException, *, run_id: UUID, **kwargs):
        self.on_tool_error(error, run_id=run_id, **kwargs)
