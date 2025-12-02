import json
import time
import threading
import os
import logging
from typing import Dict, Any, Optional, List, Tuple
from uuid import UUID
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult
from langchain_core.messages import BaseMessage

# --- 1. JSON TRACER (The "Data" Layer) ---
class PerfettoTracer:
    """
    Responsible ONLY for building the Chrome Trace Event JSON structure.
    Does not print anything to the console.
    """
    def __init__(self):
        self.trace_events = []
        self.pid = os.getpid()
        self._lock = threading.Lock()

    def add_event(self, name: str, phase: str, category: str, 
                 ts: Optional[float] = None, args: Optional[Dict] = None, 
                 tid: Optional[int] = None):
        """Records a single trace event."""
        if ts is None: 
            ts = time.time() * 1_000_000
        if tid is None: 
            tid = threading.get_ident()

        event = {
            "name": name, "cat": category, "ph": phase, "ts": ts,
            "pid": self.pid, "tid": tid, "args": args or {}
        }
        with self._lock:
            self.trace_events.append(event)

    def add_counter(self, name: str, values: Dict[str, Any], category: str = "metrics"):
        """Records a metric counter."""
        ts = time.time() * 1_000_000
        event = {
            "name": name, "cat": category, "ph": "C", "ts": ts,
            "pid": self.pid, "tid": 0, "args": values
        }
        with self._lock:
            self.trace_events.append(event)

    def save(self, filename: str):
        """Writes the buffer to disk."""
        try:
            with open(filename, 'w') as f:
                json.dump({"traceEvents": self.trace_events}, f, indent=2)
        except Exception as e:
            print(f"Error saving trace: {e}")


# --- 2. CONSOLE LOGGER (The "Presentation" Layer) ---
class PerfettoLogger:
    """
    Responsible ONLY for pretty-printing status to the console.
    Does not touch JSON or files.
    """
    def __init__(self, logger_name: str = "instrumentation"):
        self.logger = logging.getLogger(logger_name)
        # Ensure default formatting if not set externally
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s | %(levelname)-8s | %(message)s', datefmt='%H:%M:%S')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)

    def log_start(self, category: str, name: str, run_id: str, parent_id: Optional[str] = None):
        msg = f"🟢 START [{category.upper()}]: '{name}' (ID: {run_id})"
        if parent_id:
            msg += f" (Parent: {parent_id})"
        self.logger.info(msg)

    def log_end(self, category: str, name: str, tokens: int = 0):
        msg = f"🔴 END   [{category.upper()}]: '{name}'"
        if tokens > 0:
            msg += f" (Tokens: {tokens})"
        self.logger.info(msg)

    def log_map_store(self, run_id: str, name: str, category: str):
        self.logger.debug(f"   ├─ MAP STORE: '{run_id}' -> ('{name}', '{category}')")

    def log_map_retrieve(self, run_id: str, name: str):
        self.logger.debug(f"   ├─ MAP RETRIEVE: Found '{run_id}' -> '{name}'")

    def log_error(self, category: str, name: str, error: str):
        self.logger.error(f"❌ ERROR [{category.upper()}]: '{name}' -> {error}")
    
    def info(self, msg: str):
        self.logger.info(msg)


# --- 3. THE HANDLER (The "Controller" Layer) ---
class LangGraphInstrumentationHandler(BaseCallbackHandler):
    """
    Orchestrates the tracing process. 
    1. Receives Event -> 2. Updates State -> 3. Notifies Logger & Tracer
    """
    def __init__(self, tracer: PerfettoTracer, logger: PerfettoLogger):
        self.tracer = tracer
        self.console = logger # Renamed for clarity: this is the PerfettoLogger instance
        self.run_map: Dict[str, Tuple[str, str]] = {} 
        
        # Metric State
        self.total_tokens = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0

    def _extract_metadata(self, kwargs: Dict) -> Dict[str, Any]:
        """Extracts LangGraph/User metadata for the JSON trace."""
        metadata = kwargs.get("metadata", {})
        debug_args = {}
        # Keys to capture
        for key in ["thread_id", "langgraph_node", "langgraph_step", "checkpoint_id", 
                    "graph_name", "agent_name"]:
            if key in metadata:
                debug_args[key] = metadata[key]
        return debug_args
    
    def _get_llm_tokens(self, response: LLMResult) -> Tuple[int, int, int]:
        """
        Robust strategy to extract tokens from ANY provider (OpenAI, Ollama, Anthropic).
        """
        input_t, output_t, total_t = 0, 0, 0
        
        # Priority 1: Standard LangChain Usage Metadata (The "New Way")
        # This is where LangChain tries to normalize everything.
        if hasattr(response, "usage_metadata") and response.usage_metadata:
            meta = response.usage_metadata
            input_t = meta.get("input_tokens", 0)
            output_t = meta.get("output_tokens", 0)
            total_t = meta.get("total_tokens", 0)
            return input_t, output_t, total_t

        # Priority 2: LLM Output (Common for OpenAI / Legacy)
        # Your log showed: response.llm_output['token_usage']
        if response.llm_output and "token_usage" in response.llm_output:
            usage = response.llm_output["token_usage"]
            input_t = usage.get("prompt_tokens", 0)
            output_t = usage.get("completion_tokens", 0)
            total_t = usage.get("total_tokens", input_t + output_t)
            return input_t, output_t, total_t

        # Priority 3: Deep Inspection of Generations (Ollama / Anthropic)
        if response.generations:
            try:
                # We usually only care about the first generation for metrics
                first_gen = response.generations[0][0]
                
                # Check 3a: Generation Metadata (OpenAI often duplicates here)
                if hasattr(first_gen, "message") and hasattr(first_gen.message, "response_metadata"):
                    meta = first_gen.message.response_metadata
                    
                    # Case A: 'token_usage' dict exists (OpenAI / Mistral)
                    if "token_usage" in meta:
                        usage = meta["token_usage"]
                        input_t = usage.get("prompt_tokens", 0)
                        output_t = usage.get("completion_tokens", 0)
                        total_t = usage.get("total_tokens", input_t + output_t)
                    
                    # Case B: Ollama specific keys
                    elif "prompt_eval_count" in meta or "eval_count" in meta:
                        input_t = meta.get("prompt_eval_count", 0)
                        output_t = meta.get("eval_count", 0)
                        total_t = input_t + output_t
                        
                    # Case C: Anthropic / Bedrock often use 'usage'
                    elif "usage" in meta:
                        usage = meta["usage"]
                        input_t = usage.get("input_tokens", 0)
                        output_t = usage.get("output_tokens", 0)
                        total_t = input_t + output_t

            except (AttributeError, IndexError, KeyError):
                pass
                
        return input_t, output_t, total_t

    # --- EVENT HANDLERS ---

    def on_chain_start(self, serialized: Dict[str, Any], inputs: Dict[str, Any], 
                       *, run_id: UUID, parent_run_id: Optional[UUID] = None, **kwargs):
        rid = str(run_id)
        pid = str(parent_run_id) if parent_run_id else None
        metadata = kwargs.get("metadata", {})
        
        # 1. Determine Identity
        if "langgraph_node" in metadata:
            name = f"Node: {metadata['langgraph_node']}"
            category = "langgraph"
        elif "graph_name" in metadata and parent_run_id is None:
            name = f"Graph: {metadata['graph_name']}"
            category = "langgraph"
        else:
            name = kwargs.get("name") or serialized.get("name") or "Chain"
            category = "chain"

        # 2. Update State
        self.run_map[rid] = (name, category)

        # 3. Dispatch to Logger (Console)
        self.console.log_start(category, name, rid, pid)
        self.console.log_map_store(rid, name, category)

        # 4. Dispatch to Tracer (JSON)
        json_args = {**self._extract_metadata(kwargs), "run_id": rid}
        self.tracer.add_event(name, "B", category, args=json_args)

    def on_chain_end(self, outputs: Dict[str, Any], *, run_id: UUID, **kwargs):
        rid = str(run_id)
        if rid in self.run_map:
            name, category = self.run_map[rid]
            # Dispatch
            self.console.log_map_retrieve(rid, name)
            self.console.log_end(category, name)
            self.tracer.add_event(name, "E", category)
        else:
            self.tracer.add_event("UnknownChain", "E", category="chain")

    def on_chain_error(self, error: BaseException, *, run_id: UUID, **kwargs):
        rid = str(run_id)
        if rid in self.run_map:
            name, category = self.run_map[rid]
            self.console.log_error(category, name, str(error))
            self.tracer.add_event(name, "E", category, args={"error": str(error)})

    def on_chat_model_start(self, serialized: Dict[str, Any], messages: List[List[BaseMessage]], 
                           *, run_id: UUID, **kwargs):
        rid = str(run_id)
        params = kwargs.get("invocation_params", {})
        model = (params.get("model") or 
             params.get("model_name") or 
             serialized.get("kwargs", {}).get("model") or 
             "ChatModel")        
        name = f"LLM: {model}"
        category = "llm"
        
        self.run_map[rid] = (name, category)
        
        self.console.log_start(category, name, rid)
        self.tracer.add_event(name, "B", category, args={"model": model})

    def on_llm_end(self, response: LLMResult, *, run_id: UUID, **kwargs):
        rid = str(run_id)
        name, category = self.run_map.get(rid, ("ChatModel", "llm"))
        
        input_t, output_t, total_t = self._get_llm_tokens(response)

        # Metrics
        if total_t > 0:
            self.total_tokens += total_t
            self.prompt_tokens += input_t
            self.completion_tokens += output_t
            self.tracer.add_counter("Token Usage", {
                "Total": self.total_tokens,
                "Input": self.prompt_tokens,
                "Output": self.completion_tokens
            })

        self.console.log_end(category, name, tokens=total_t)
        self.tracer.add_event(name, "E", category, args={
            "tokens": total_t, "input": input_t, "output": output_t
        })

    def on_llm_error(self, error: BaseException, *, run_id: UUID, **kwargs):
        rid = str(run_id)
        if rid in self.run_map:
            name, category = self.run_map[rid]
            self.console.log_error(category, name, str(error))
            self.tracer.add_event(name, "E", category, args={"error": str(error)})

    def on_tool_start(self, serialized: Dict[str, Any], input_str: str, *, run_id: UUID, **kwargs):
        rid = str(run_id)
        name = f"Tool: {serialized.get('name') or 'Unknown'}"
        category = "tool"
        
        self.run_map[rid] = (name, category)
        
        self.console.log_start(category, name, rid)
        self.tracer.add_event(name, "B", category)

    def on_tool_end(self, output: Any, *, run_id: UUID, **kwargs):
        rid = str(run_id)
        if rid in self.run_map:
            name, category = self.run_map[rid]
            self.console.log_end(category, name)
            self.tracer.add_event(name, "E", category)

    def on_tool_error(self, error: BaseException, *, run_id: UUID, **kwargs):
        rid = str(run_id)
        if rid in self.run_map:
            name, category = self.run_map[rid]
            self.console.log_error(category, name, str(error))
            self.tracer.add_event(name, "E", category, args={"error": str(error)})