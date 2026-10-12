"""Import and prompt checks for the graphs this package ships."""

from __future__ import annotations

import tomllib
from pathlib import Path

from langchain_core.messages import AIMessage, HumanMessage

from agents import __version__
from agents.graphs.react.graph import ReactAgent
from agents.graphs.simple.graph import SimpleChatAgent

_ROOT = Path(__file__).resolve().parents[1]


def test_version_matches_pyproject() -> None:
    with (_ROOT / "pyproject.toml").open("rb") as handle:
        project_version = tomllib.load(handle)["project"]["version"]
    assert __version__ == project_version


def test_graphs_import_and_load_system_prompt() -> None:
    react = ReactAgent()
    simple = SimpleChatAgent()
    assert react.name == "react"
    assert simple.name == "simple"
    assert react.prompts["system"].strip()
    assert simple.prompts["system"].strip()


def test_format_history_empty() -> None:
    agent = SimpleChatAgent()
    assert agent.format_history([], None) is None
    assert agent.format_history([], "   ") is None


def test_format_history_summary_and_roles() -> None:
    text = SimpleChatAgent().format_history(
        [
            HumanMessage(content="hello"),
            {"role": "assistant", "content": "hi"},
            AIMessage(content="again"),
        ],
        "earlier talk",
    )
    assert text is not None
    assert "Previous conversation summary:\nearlier talk\n" in text
    assert "User: hello" in text
    assert "Assistant: hi" in text
    assert "Assistant: again" in text


def test_instruction_text_prepends_history() -> None:
    agent = SimpleChatAgent()
    body = agent.prompts["system"].strip()
    text = agent.instruction_text(
        history=[{"role": "user", "content": "ping"}],
        summary=None,
    )
    assert text is not None
    assert text.startswith("User: ping\n\n")
    assert text.endswith(body)
