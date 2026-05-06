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

logger = logging.getLogger(__name__)


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


class AgentGraph:
    """Base class for pluggable agent graphs.

    At construction, :attr:`prompts` is filled from ``prompts.yaml`` next to the
    concrete class's ``graph.py`` (same keys as in the file, typically a string
    field ``system`` for the lead-in message).

    Inherit and implement ``compile()``.  Use the helper methods for free
    node/tool latency tracking and error handling.

    Only depends on langchain-core + langgraph + PyYAML.  No backend imports.

    Subclasses that override ``__init__`` must call ``super().__init__()`` so
    :attr:`prompts` is populated.
    """

    name: str = "base"

    prompts: Dict[str, Any]

    def __init__(self) -> None:
        self.prompts = self._load_prompts_from_yaml()

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

    def instruction_text(self, *, conversation_prefix: Optional[str] = None) -> Optional[str]:
        """Lead-in message: ``prompts['system']`` plus optional *conversation_prefix*."""
        raw = self.prompts.get("system")
        body = str(raw).strip() if raw is not None else None
        if body == "":
            body = None
        pfx = str(conversation_prefix).strip() if conversation_prefix else ""
        if pfx:
            if body:
                return f"{pfx}{body}"
            return pfx or None
        return body

    def compile(
        self,
        *,
        llm: BaseChatModel,
        tools: List[BaseTool],
        checkpointer: Any,
        conversation_prefix: Optional[str] = None,
    ) -> CompiledStateGraph:
        """Build and return the compiled StateGraph.  Override in subclass."""
        raise NotImplementedError

    # ── Helper: instrumented tools node ────────────────────────────────────────

    def make_tools_node(self, tools: List[BaseTool]):
        """Create a tools node with per-tool latency tracking and error handling.

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

    # ── Helper: timed node wrapper ─────────────────────────────────────────────

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
