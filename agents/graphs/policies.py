"""LangGraph fault-tolerance policies shared across agent graphs.

Requires ``langgraph>=1.2`` for :class:`~langgraph.types.TimeoutPolicy` and
node-level ``error_handler`` support.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Optional

from langgraph.errors import NodeError
from langgraph.types import Command, RetryPolicy, TimeoutPolicy

logger = logging.getLogger(__name__)

# Default LLM attempt caps — backends can override via ``compile(llm_run_timeout=…)``.
DEFAULT_LLM_RUN_TIMEOUT = 120.0
DEFAULT_LLM_IDLE_TIMEOUT = 30.0

# Transient failures on external tool/MCP calls; not used on LLM nodes.
TOOL_RETRY = RetryPolicy(
    max_attempts=5,
    initial_interval=1.0,
    backoff_factor=2.0,
    max_interval=10.0,
    jitter=True,
)


def build_llm_timeout_policy(
    *,
    run_timeout: Optional[float] = DEFAULT_LLM_RUN_TIMEOUT,
    idle_timeout: Optional[float] = DEFAULT_LLM_IDLE_TIMEOUT,
) -> Optional[TimeoutPolicy]:
    """Build a :class:`TimeoutPolicy` for an LLM node.

    Returns ``None`` only when both limits are ``None`` (no node-level timeout).
    """
    if run_timeout is None and idle_timeout is None:
        return None
    return TimeoutPolicy(run_timeout=run_timeout, idle_timeout=idle_timeout)


def make_llm_fallback_error_handler(
    *,
    node: str,
    has_fallback: bool,
    log: logging.Logger = logger,
) -> Callable[[Any, NodeError], Command]:
    """Return an ``error_handler`` that re-runs *node* with the fallback model.

    Fires when the LLM node fails and no ``retry_policy`` is configured on that
    node (the ``TimeoutPolicy`` and any other exception go straight to the
    handler).  If the fallback was already attempted on this run or no fallback
    model was supplied, the original exception is re-raised.
    """

    def handler(state: Any, error: NodeError) -> Command:
        if not has_fallback or state.get("use_fallback_llm"):
            raise error.error
        log.warning(
            "Node %s failed (%s: %s), retrying with fallback model",
            error.node,
            type(error.error).__name__,
            error.error,
        )
        return Command(update={"use_fallback_llm": True}, goto=node)

    return handler


def llm_node_add_kwargs(
    *,
    node: str,
    has_fallback: bool,
    llm_run_timeout: Optional[float] = DEFAULT_LLM_RUN_TIMEOUT,
    llm_idle_timeout: Optional[float] = DEFAULT_LLM_IDLE_TIMEOUT,
) -> dict[str, Any]:
    """``add_node`` keyword args for a fault-tolerant LLM node."""
    kwargs: dict[str, Any] = {
        "error_handler": make_llm_fallback_error_handler(
            node=node,
            has_fallback=has_fallback,
        ),
    }
    timeout = build_llm_timeout_policy(
        run_timeout=llm_run_timeout,
        idle_timeout=llm_idle_timeout,
    )
    if timeout is not None:
        kwargs["timeout"] = timeout
    return kwargs
