"""The react prompt describes eve_retrieval_retrieve only when that tool is bound."""

import subprocess
from pathlib import Path

import yaml

from agents.graphs.react.graph import ReactAgent

RETRIEVE = "eve_retrieval_retrieve"


def _prompt_before_tool_sections() -> str:
    """The react system prompt as it was before tool sections (commit bbc2fad)."""
    raw = subprocess.run(
        ["git", "show", "bbc2fad:agents/graphs/react/prompts.yaml"],
        cwd=Path(__file__).resolve().parents[1],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return str(yaml.safe_load(raw)["system"]).strip()


def test_bound_retrieval_renders_the_prompt_unchanged():
    agent = ReactAgent()

    assert agent.instruction_text(tool_names=[RETRIEVE, "other"]) == (
        _prompt_before_tool_sections()
    )


def test_unknown_tools_keep_every_section():
    # Callers that do not pass tool names (the backend trace) see the full prompt.
    assert ReactAgent().instruction_text() == _prompt_before_tool_sections()


def test_no_retrieval_tool_means_no_mention_of_it():
    text = ReactAgent().instruction_text(tool_names=[])

    assert RETRIEVE not in text
    assert "[[" not in text
    assert "\n\n\n" not in text
    # The rest of the prompt is still there.
    assert "## Your Identity" in text
    assert "## Response Format Guidelines" in text


def test_other_tools_only_still_drop_the_retrieval_sections():
    text = ReactAgent().instruction_text(tool_names=["effis_search"])

    assert RETRIEVE not in text
    assert "## Core Operating Principles" in text


def test_history_prefix_is_kept_with_sections_dropped():
    text = ReactAgent().instruction_text(
        history=[{"role": "user", "content": "hello"}], tool_names=[]
    )

    assert text.startswith("User: hello")
    assert RETRIEVE not in text
