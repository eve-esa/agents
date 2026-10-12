# eve-esa-agents — agent notes

Pip-installable LangGraph agent definitions. The importable package is `agents`
(flat layout at repo root) — not a nested folder despite the `agents/agents`
checkout path.

## Setup
- Plain pip + venv (no lockfile). Dev install: `pip install -e ".[dev]"`

## Commands
- Test: `pytest` (single: `pytest tests/test_graphs.py::test_version_matches_pyproject`)
- No linter / formatter / typecheck is configured — don't invent one.

## Release
- A push to `main` dispatches backend `deploy-dev` on backend `main` (`.github/workflows/notify-backend.yml`). That build installs this commit. That is dev.
- release-please (`.github/workflows/release.yml`) cuts `vX.Y.Z` from Conventional Commit pull request titles. The release uploads the sdist and wheel, and opens a backend pull request titled `fix(deps): pin agents to vX.Y.Z` that writes the tag to `agents.ref`.
- Staging and prod install that tag, when the backend promotes the image built from it. Production is still one approval on the backend's promote-prod workflow.
- `BACKEND_DISPATCH_TOKEN` on this repo must be able to dispatch workflows and open pull requests on `eve-esa/backend`.

## Conventions & constraints
- Imports inside `agents/graphs/` MUST be relative so that subtree can be developed as its own standalone repo.
- Dependency pins in `pyproject.toml` are deliberately conservative (e.g. `langgraph>=1.2,<2`) so installing into a backend doesn't pull a mismatched langgraph/langchain-core. Do not loosen them casually.
- Each graph ships `prompts.yaml` next to `graph.py`, declared as `package-data` in `pyproject.toml` — add new ones there or they won't ship in the wheel.
- The version lives in `pyproject.toml` and on the `x-release-please-version` line in `agents/__init__.py`. release-please updates both from `.release-please-manifest.json`. Do not bump them by hand. The `pyproject.toml` updater rewrites that file and drops comments, so do not put notes there that have to survive a release.
