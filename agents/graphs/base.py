"""AgentGraph base class — the contract for pluggable agent graphs.

Only depends on langchain-core + langgraph + PyYAML + standard library.
No backend (src.*) imports. Safe to use in standalone scripts/notebooks.
"""

import inspect
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import ToolMessage
from langchain_core.tools import BaseTool
from langgraph.graph import MessagesState
from langgraph.graph.state import CompiledStateGraph

from .policies import DEFAULT_LLM_IDLE_TIMEOUT, DEFAULT_LLM_RUN_TIMEOUT

logger = logging.getLogger(__name__)


class AgentMessagesState(MessagesState):
    """Messages graph state for pluggable agent graphs.

    Extends :class:`~langgraph.graph.MessagesState` (which provides the
    ``messages`` list) with no additional fields by default.  Kept as an
    explicit subclass so that:

    - All node functions share a single type annotation that can be
      extended in-place without touching every ``add_node`` call.
    - Future per-graph state fields (e.g. retrieved-doc metadata, turn
      counters) can be added here rather than requiring a new class.

    Recovery routing is structural (via the ``agent_fallback`` node) rather
    than flag-based, so no ``use_fallback_llm`` field is needed in state.
    """


# ─── Standalone MCP interceptor (no backend dependencies) ─────────────────────


class LatencyInterceptor:
    """MCP tool-call interceptor that tracks latency via standard logging.

    Follows the ``ToolCallInterceptor`` protocol from ``langchain-mcp-adapters``.
    No backend dependencies — safe to use in standalone scripts/notebooks.

    Usage with ``MultiServerMCPClient``::

        client = MultiServerMCPClient(
            connections,
            tool_interceptors=[LatencyInterceptor()],
        )
    """

    async def __call__(self, request: Any, handler: Any) -> Any:
        tool_name = getattr(request, "name", "unknown")
        server_name = getattr(request, "server_name", "unknown")
        start = time.perf_counter()

        try:
            result = await handler(request)
            elapsed = time.perf_counter() - start
            logger.info(
                "MCP tool %s (server=%s) completed in %.3fs",
                tool_name,
                server_name,
                elapsed,
            )
            return result
        except Exception as exc:
            elapsed = time.perf_counter() - start
            logger.error(
                "MCP tool %s (server=%s) failed after %.3fs: %s",
                tool_name,
                server_name,
                elapsed,
                exc,
            )
            raise


# ─── AgentGraph base class ────────────────────────────────────────────────────

# Role labels used when serialising history turns to plain text.
_ROLE_LABELS: Dict[str, str] = {
    "human": "User",
    "humanmessage": "User",
    "user": "User",
    "ai": "Assistant",
    "aimessage": "Assistant",
    "assistant": "Assistant",
    "system": "System",
    "systemmessage": "System",
    "tool": "Tool",
    "toolmessage": "Tool",
}


class AgentGraph:
    """Base class for pluggable agent graphs.

    At construction, :attr:`prompts` is filled from ``prompts.yaml`` next to the
    concrete class's ``graph.py`` (keys from the file, typically ``system`` for the
    lead-in instruction).

    ``compile`` receives the raw **conversation history** (list of LangChain messages
    or ``{"role": …, "content": …}`` dicts) and the optional **summary** string from
    the backend.  The base :meth:`format_history` serialises them to a plain-text
    prefix that is prepended to ``prompts["system"]`` inside :meth:`instruction_text`.
    Subclasses can override :meth:`format_history` to change that serialisation (e.g.
    keep messages as structured objects rather than text).

    Only depends on langchain-core + langgraph + PyYAML.  No backend imports.

    Subclasses that override ``__init__`` must call ``super().__init__()`` so
    :attr:`prompts` is populated.
    """

    name: str = "base"

    prompts: Dict[str, Any]

    def __init__(self) -> None:
        self.prompts = self._load_prompts_from_yaml()

    # ── Prompts YAML ──────────────────────────────────────────────────────────

    def _prompts_yaml_path(self) -> Path:
        """``prompts.yaml`` in the same directory as the module defining the graph class."""
        mod = inspect.getmodule(type(self))
        if mod is None or not getattr(mod, "__file__", None):
            raise RuntimeError(
                f"Cannot locate prompts.yaml for {type(self).__qualname__}: "
                "defining module has no __file__"
            )
        return Path(mod.__file__).resolve().parent / "prompts.yaml"

    def _load_prompts_from_yaml(self) -> Dict[str, Any]:
        path = self._prompts_yaml_path()
        if not path.is_file():
            logger.warning(
                "Missing prompts.yaml for graph %r (expected %s)",
                type(self).__name__,
                path,
            )
            return {}
        try:
            import yaml

            data = yaml.safe_load(path.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.error("Failed to read %s: %s", path, exc, exc_info=True)
            return {}
        if not isinstance(data, dict):
            return {}
        return dict(data)

    # ── History formatting ────────────────────────────────────────────────────

    def format_history(
        self,
        history: List[Any],
        summary: Optional[str],
    ) -> Optional[str]:
        """Serialise conversation history and summary to a plain-text prefix string.

        The base implementation:

        1. Emits ``"Previous conversation summary:\\n{summary}\\n"`` when *summary* is set.
        2. Appends each message as ``"{Role}: {content}"`` lines.

        Override in a subclass to use a different format (e.g. keep messages as
        structured objects, or suppress the summary, etc.).

        Returns ``None`` when both inputs are empty.
        """
        parts: List[str] = []

        if summary and str(summary).strip():
            parts.append(
                f"Previous conversation summary:\n{summary.strip()}\n"
                "Please continue the conversation using this summary as context."
            )

        if history:
            turns: List[str] = []
            for msg in history:
                # LangChain message objects
                if hasattr(msg, "content"):
                    raw_role = type(msg).__name__
                    content = msg.content
                # plain dicts {"role": …, "content": …}
                elif isinstance(msg, dict):
                    raw_role = str(msg.get("role", "unknown"))
                    content = msg.get("content", "")
                else:
                    raw_role = "unknown"
                    content = str(msg)

                if isinstance(content, list):
                    # multi-part content (e.g. vision models)
                    content = " ".join(
                        c.get("text", "") if isinstance(c, dict) else str(c)
                        for c in content
                    )

                label = _ROLE_LABELS.get(raw_role.lower(), raw_role.capitalize())
                turns.append(f"{label}: {str(content).strip()}")

            if turns:
                parts.append("\n".join(turns))

        if not parts:
            return None
        return "\n\n".join(parts)

    # ── Instruction builder ───────────────────────────────────────────────────

    def instruction_text(
        self,
        history: Optional[List[Any]] = None,
        summary: Optional[str] = None,
    ) -> Optional[str]:
        """Build the full lead-in instruction: history prefix + ``prompts['system']``.

        *history* and *summary* are serialised by :meth:`format_history` and
        prepended to the YAML ``system`` value.  Either or both may be omitted.
        """
        raw = self.prompts.get("system")
        body = str(raw).strip() if raw else None
        if not body:
            body = None

        prefix = self.format_history(history or [], summary)

        if prefix and body:
            return f"{prefix}\n\n{body}"
        if prefix:
            return prefix
        return body

    # ── Compile (override per subclass) ──────────────────────────────────────

    def compile(
        self,
        *,
        llm: BaseChatModel,
        tools: List[BaseTool],
        checkpointer: Any,
        history: Optional[List[Any]] = None,
        summary: Optional[str] = None,
        fallback_llm: Optional[BaseChatModel] = None,
        llm_run_timeout: Optional[float] = DEFAULT_LLM_RUN_TIMEOUT,
        llm_idle_timeout: Optional[float] = DEFAULT_LLM_IDLE_TIMEOUT,
        streaming: bool = True,
        **kwargs: Any,
    ) -> CompiledStateGraph:
        """Build and return the compiled StateGraph.  Override in subclass.

        *fallback_llm* — optional secondary model; on LLM node failure the graph
        ``error_handler`` re-runs the node with this binding (see
        :mod:`agents.graphs.policies`).

        *llm_run_timeout* / *llm_idle_timeout* — per-attempt caps for LLM nodes
        (``TimeoutPolicy``).  Pass ``None`` for both to disable timeouts.
        ``llm_idle_timeout`` applies only when *streaming* is ``True``; non-streaming
        graphs rely on ``llm_run_timeout`` alone.
        """
        raise NotImplementedError

    # ── Helper: instrumented tools node ──────────────────────────────────────

    def make_tools_node(self, tools: List[BaseTool]):
        """Create a tools node with per-tool latency tracking and error handling.

        Tool failures are caught and returned as ``ToolMessage`` content so the
        agent can recover within the ReAct loop.  Do **not** attach a node-level
        ``retry_policy`` here: the node never re-raises (so it would never fire)
        and a node-level retry would re-invoke every tool call in the turn.

        Usage::

            builder.add_node("tools", self.make_tools_node(tools))
        """
        tool_map = {t.name: t for t in tools}

        async def tools_node(state: MessagesState):
            last = state["messages"][-1]
            results: List[ToolMessage] = []
            for tc in getattr(last, "tool_calls", []):
                name, args, call_id = tc["name"], tc["args"], tc["id"]
                tool = tool_map.get(name)
                start = time.perf_counter()
                if tool is None:
                    result = f"Unknown tool: {name}"
                    logger.warning("Tool %s not found in tool_map", name)
                else:
                    try:
                        result = await tool.ainvoke(args)
                        elapsed = time.perf_counter() - start
                        logger.info(
                            "Tool %s completed in %.3fs (%d chars)",
                            name,
                            elapsed,
                            len(str(result)),
                        )
                    except Exception as exc:
                        elapsed = time.perf_counter() - start
                        logger.error(
                            "Tool %s failed after %.3fs: %s", name, elapsed, exc
                        )
                        result = f"Tool error: {exc}"
                results.append(
                    ToolMessage(content=str(result), tool_call_id=call_id, name=name)
                )
            return {"messages": results}

        return tools_node

    # ── Helper: timed node wrapper ────────────────────────────────────────────

    def timed_node(self, node_name: str, fn):
        """Wrap any async node function with latency logging.

        Usage::

            builder.add_node("agent", self.timed_node("agent", my_agent_fn))
        """

        async def wrapper(state):
            start = time.perf_counter()
            result = await fn(state)
            elapsed = time.perf_counter() - start
            logger.info("Node '%s' completed in %.3fs", node_name, elapsed)
            return result

        wrapper.__name__ = node_name
        return wrapper
