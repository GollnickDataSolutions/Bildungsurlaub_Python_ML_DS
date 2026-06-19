# AGENTS.md — Python ML/DS Kursmaterial

## Project

German-language course "Python für Machine Learning and Data Science" (Bildungsurlaub).  
Educational material — not a library or app. No tests, no CI/CD, no lint/typecheck config.

## Setup

- Python `>=3.12` (via `.python-version`)
- Package manager: **uv** (`pip install uv && uv sync`) — lockfile `uv.lock` is the source of truth
- Alternative: `python -m venv venv && pip install -r requirements.txt` (no `requirements.txt` exists; use `uv export` or `pyproject.toml` directly)
- `.venv/` is gitignored

## Structure

- `NNN_TopicName/` — numbered course modules, each with PDF slides + `.py` scripts + `.ipynb` notebooks
- `our_scripts/` — student/workshop Python scripts and CSV datasets
- `change_point/` — change point detection experiment
- `300_VibeCoding/` — AI-assisted coding module
- `_archive/` — gitignored (stale content)
- `.opencode/skills/` — OpenCode skills (explain-code, skill-creator)

## Conventions

- Scripts are standalone (`python script.py`) or Jupyter notebooks; no package imports across modules
- Don't add dependencies without updating `pyproject.toml` and running `uv lock`
- German is the course language; code comments, docstrings, and variable names are often English
- No test framework — verify scripts by running them manually
