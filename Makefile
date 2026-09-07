# Makefile for the "call me maybe" project.
#
# The reviewer / moulinette run only `uv sync` then `uv run python -m src ...`.
# These targets wrap that, plus the lint/clean rules required by the subject.

# Extra arguments forwarded to the program, e.g.:
#   make run ARGS="--input data/input/function_calling_tests.json"
ARGS ?=

MYPY_FLAGS := --warn-return-any --warn-unused-ignores --ignore-missing-imports \
              --disallow-untyped-defs --check-untyped-defs

# ── Local environment (42 workstation only) ─────────────────────────────
# The home partition is ~2 GB: too small for the torch install plus the
# 1.5 GB model. When the scratch partitions exist, `_setup`:
#   * points UV_CACHE_DIR / HF_HOME at them (exported to every recipe), and
#   * makes .venv a symlink to a venv on the fast local partition, so that a
#     bare `uv sync` / `uv run` (what the corrector uses) also stays off the
#     home partition -- no reliance on `make`.
# A machine without these partitions (the moulinette) is untouched: .venv is
# a normal directory from a normal `uv sync`.
VENV_TARGET := $(HOME)/goinfre/call_me_maybe/venv

ifneq ($(wildcard $(HOME)/goinfre/.),)
export UV_CACHE_DIR := $(HOME)/goinfre/call_me_maybe/uv-cache
ON_SCRATCH := 1
endif
ifneq ($(wildcard $(HOME)/Sgoinfre/hf_cache/.),)
export HF_HOME := $(HOME)/Sgoinfre/hf_cache
endif

.PHONY: all _setup install run debug lint lint-strict test clean fclean re

all: install

_setup:
ifdef ON_SCRATCH
	@mkdir -p $(VENV_TARGET) $(HOME)/goinfre/call_me_maybe/uv-cache
	@if [ ! -L .venv ]; then rm -rf .venv && ln -s $(VENV_TARGET) .venv; fi
endif

# Install dependencies with uv.
install: _setup
	uv sync

# Run the project. Default paths: data/input/ -> data/output/.
run: _setup
	uv run python -m src $(ARGS)

# Run the project under the standard-library debugger (pdb).
debug: _setup
	uv run python -m pdb -m src $(ARGS)

# flake8 + mypy on our code. llm_sdk (provided) and .venv are excluded via
# .flake8 and pyproject.toml so the subject's `flake8 .` / `mypy .` still work.
lint: _setup
	uv run flake8 .
	uv run mypy . $(MYPY_FLAGS)

lint-strict: _setup
	uv run flake8 .
	uv run mypy . --strict

# Run the test suite. Exit code 5 (pytest found no tests) is tolerated.
test: _setup
	uv run pytest || [ $$? -eq 5 ]

# Remove Python and tooling caches.
clean:
	find . -type d -name '__pycache__'   -prune -exec rm -rf {} +
	find . -type d -name '.mypy_cache'   -prune -exec rm -rf {} +
	find . -type d -name '.pytest_cache' -prune -exec rm -rf {} +
	find . -type d -name '.ruff_cache'   -prune -exec rm -rf {} +
	find . -type f -name '.dmypy.json'   -delete

# clean + drop the virtual environment and generated output.
fclean: clean
	rm -rf .venv $(VENV_TARGET)
	rm -rf data/output

re: fclean install
