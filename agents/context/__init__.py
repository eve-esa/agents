"""Saved tool results, automatic memory, and bounded prompts for agent graphs."""

from .builder import ContextBuilder
from .config import ContextSettings
from .memory import WorkingMemory
from .results import ResultReader
from .tools import recovery_tools

CONTEXT_API_VERSION = 1


class AgentContext:
    def __init__(self, results, memory_repository, *, settings=None, on_source=None):
        self.settings = settings or ContextSettings()
        self.results = results
        self.memory = WorkingMemory(memory_repository, results, self.settings)
        self.on_source = on_source

    def tools(self):
        return recovery_tools(self.results, self.memory, self.on_source)

    def with_recovery_tools(self, tools):
        recovery = self.tools()
        reserved = {tool.name for tool in recovery}
        if any(tool.name in reserved for tool in tools):
            raise ValueError(
                "A supplied tool conflicts with a reserved context recovery tool name"
            )
        return [*tools, *recovery]

    def bind(self, instruction, tools, llm):
        return ContextBuilder(
            self.results, self.memory, self.settings, instruction, tools, llm
        )


__all__ = [
    "CONTEXT_API_VERSION",
    "AgentContext",
    "ContextSettings",
    "ResultReader",
    "WorkingMemory",
]
