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

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult
from langchain_core.messages import BaseMessage


# =============================================================================
# TRACER
# =============================================================================

class PerfettoTracer:
    def __init__(self):
        self.trace_events: List[Dict[str, Any]] = []
        self.pid = os.getpid()
        self._lock = threading.Lock()

    @staticmethod
    def _now_us() -> int:
        return time.perf_counter_ns() // 1000

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

        with self._lock:
            self.trace_events.append(ev)

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
        with self._lock:
            self.trace_events.append(ev)

    def add_counter(
        self,
        name: str,
        values: Dict[str, Any],
        ts_us: Optional[int] = None,
        tid: Optional[int] = None,
    ):
        self.add_event(name=name, phase="C", category="metrics", ts_us=ts_us, tid=tid, args=values)

    def set_thread_name(self, tid: int, name: str):
        # Track/thread naming metadata
        self.add_event(
            name="thread_name",
            phase="M",
            category="__metadata",
            tid=tid,
            args={"name": name},
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

    def save(self, filename: str):
        with self._lock:
            snapshot = list(self.trace_events)
        with open(filename, "w") as f:
            json.dump({"traceEvents": snapshot}, f, indent=2, default=str)


# =============================================================================
# LOGGER
# =============================================================================

class PerfettoLogger:
    def __init__(self, logger_name: str = "instrumentation"):
        self.logger = logging.getLogger(logger_name)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter("%(asctime)s | %(message)s", datefmt="%H:%M:%S")
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)
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
    Key changes vs your original:
      1) Tool calls get a stable *virtual tid* derived from tool_call_id, so overlapping tools render on separate tracks.
      2) Async callback variants (aon_*) are implemented so async tool calls are traced properly.
      3) tool_call_id is read directly from kwargs['tool_call_id'] (as you specified).
    """

    # Keep virtual tids away from real thread ids
    _TOOL_TID_OFFSET = 10_000_000

    def __init__(self, tracer: PerfettoTracer, logger: PerfettoLogger):
        super().__init__()
        self.tracer = tracer
        self.console = logger

        self.run_map: Dict[str, RunInfo] = {}
        self._run_lock = threading.Lock()  # REQUIRED

        self._named_tids: Set[int] = set()
        self._named_tids_lock = threading.Lock()

        self._flow_lock = threading.Lock()
        self._flow_counter = 1

        self.token_counts = {"total": 0, "input": 0, "output": 0}

        self.seen_message_ids: Set[str] = set()
        self._msg_lock = threading.Lock()

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
        # Deterministic, stable int track id derived from tool_call_id
        h = zlib.crc32(tool_call_id.encode("utf-8")) & 0xFFFFFFFF
        return self._TOOL_TID_OFFSET + int(h)

    def _extract_metadata(self, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        meta = kwargs.get("metadata", {}) or {}
        keep_keys = {"thread_id", "langgraph_node", "checkpoint_id", "graph_name"}
        return {k: meta[k] for k in keep_keys if k in meta}

    def _sanitize_state(self, data: Any) -> Any:
        if isinstance(data, dict):
            new_data = {}
            for k, v in data.items():
                if k == "messages" and isinstance(v, list):
                    new_data[k] = [self._process_single_message(m) for m in v]
                else:
                    new_data[k] = self._sanitize_state(v)
            return new_data
        if isinstance(data, list):
            return [self._sanitize_state(x) for x in data]
        return data

    def _process_single_message(self, message: Any) -> Any:
        if not isinstance(message, BaseMessage):
            return str(message)

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

        if msg_key is not None:
            with self._msg_lock:
                if msg_key in self.seen_message_ids:
                    ref = {"id": msg_id}
                    if hasattr(message, "type"):
                        ref["type"] = getattr(message, "type")
                    return ref
                self.seen_message_ids.add(msg_key)

        return msg_data

    def _start_run(
        self,
        run_id: UUID,
        name: str,
        category: str,
        args: Optional[Dict[str, Any]],
        parent_run_id: Optional[UUID],
        tid_override: Optional[int] = None,  # <-- NEW
    ):
        rid = str(run_id)
        start_ts = self._now_us()

        # IMPORTANT: use the override tid if provided (virtual tool tracks)
        tid = tid_override if tid_override is not None else threading.get_ident()

        self._ensure_thread_named(tid)

        parent_key = str(parent_run_id) if parent_run_id is not None else None

        # Flow link parent -> child (across threads/tracks)
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
                self.tracer.add_flow_start(flow_id, start_ts, parent_info.tid, flow_args)
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
            marker["real_tid"] = threading.get_ident()
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
            # still emit something visible
            ts = self._now_us()
            args = end_args or {}
            if error:
                args = dict(args)
                args["error"] = error
            self.tracer.add_complete_event("Unknown", "unknown", ts, 1, threading.get_ident(), args=args)
            return

        end_ts = self._now_us()
        dur = max(1, end_ts - info.start_ts_us)

        args: Dict[str, Any] = {}
        if end_args:
            args.update(end_args)
        if error:
            args["error"] = error

        args["parent_run_id"] = info.parent_run_id
        args["real_tid_start"] = info.tid
        args["real_tid_end"] = threading.get_ident()

        # IMPORTANT: end on the tid stored at start (virtual tool track works even if callback fires elsewhere)
        self.tracer.add_complete_event(info.name, info.category, info.start_ts_us, dur, info.tid, args=args)

        if error:
            self.console.log_step("ERROR", info.category, info.name, error)
        else:
            self.console.log_step("END", info.category, info.name)

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
                sanitized_input = self._sanitize_state(inputs)
                self.console.log_state(node_name, f"Input -> {str(sanitized_input)[:100]}...")
            elif "graph_name" in metadata:
                name = f"Graph: {metadata['graph_name']}"
                category = "graph_root"
                sanitized_input = self._sanitize_state(inputs)
            else:
                sanitized_input = self._sanitize_state(inputs)
        except Exception as e:
            sanitized_input = {"instrumentation_error": str(e)}
            self.console.log_step("ERROR", "instrumentation", "on_chain_start", str(e))

        trace_args = {**self._extract_metadata(kwargs), "inputs": sanitized_input}
        if parent_run_id is not None:
            trace_args["parent_run_id"] = str(parent_run_id)

        self._start_run(run_id, name, category, trace_args, parent_run_id)

    def on_chain_end(self, outputs: Dict[str, Any], *, run_id: UUID, **kwargs):
        sanitized_output = self._sanitize_state(outputs)
        self._end_run(run_id, end_args={"outputs": sanitized_output})

    def on_chain_error(self, error: BaseException, *, run_id: UUID, **kwargs):
        self._end_run(run_id, error=str(error))

    # -------------------------------------------------------------------------
    # CHAIN / NODE (async) — delegates to the same logic
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
    # TOOL (sync) — NEW: tool_call_id -> virtual tid track
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

        # As requested: tool_call_id is in kwargs['tool_call_id']
        tool_call_id = kwargs.get("tool_call_id")
        virtual_tid: Optional[int] = None

        if tool_call_id:
            tool_call_id = str(tool_call_id)
            virtual_tid = self._tool_virtual_tid(tool_call_id)
            # Name the virtual tool track nicely once
            self._ensure_thread_named(virtual_tid, f"tool:{tool_name}:{tool_call_id}")

        self._start_run(
            run_id,
            f"Tool: {tool_name}",
            "tool",
            {"input": input_str, "tool_call_id": tool_call_id},
            parent_run_id,
            tid_override=virtual_tid,
        )

    def on_tool_end(self, output: Any, *, run_id: UUID, **kwargs):
        safe_output = self._sanitize_state(output) if isinstance(output, (dict, list)) else str(output)
        self._end_run(run_id, end_args={"output": safe_output})

    def on_tool_error(self, error: BaseException, *, run_id: UUID, **kwargs):
        self._end_run(run_id, error=str(error))

    # -------------------------------------------------------------------------
    # TOOL (async) — IMPORTANT for LangGraph async tool calls
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
        # This is very lightweight; calling sync logic directly is fine.
        self.on_tool_start(serialized, input_str, run_id=run_id, parent_run_id=parent_run_id, **kwargs)

    async def aon_tool_end(self, output: Any, *, run_id: UUID, **kwargs):
        self.on_tool_end(output, run_id=run_id, **kwargs)

    async def aon_tool_error(self, error: BaseException, *, run_id: UUID, **kwargs):
        self.on_tool_error(error, run_id=run_id, **kwargs)
