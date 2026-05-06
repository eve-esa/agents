"""Standalone agent graph code (no backend imports).

Contains ``base`` (AgentGraph, LatencyInterceptor), ``utils``, and graph
implementations in sub-packages (e.g. ``react/``). Each graph package may include
``prompts.yaml`` next to ``graph.py``; keys populate ``AgentGraph.prompts`` at init.

All imports **inside** this directory use relative imports (``from ..base``,
``from .graph``, etc.) so this tree can live in its own repository.

Backend integration in a consuming app lives outside this package (e.g. a
``core`` module that wires MCP, persistence, and env).
"""
