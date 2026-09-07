# Makefile for the "call me maybe" project.
#
# The reviewer and the moulinette only run `uv sync` followed by
# `uv run python -m src ...`. These targets are convenience wrappers around
# that, plus the lint/clean rules required by the subject.

# Extra arguments forwarded to the program, e.g.:
#   make run ARGS="--input data/input/function_calling_tests.json"
ARGS ?=

MYPY_FLAGS := --warn-return-any --warn-unused-ignores --ignore-missing-imports \
              --disallow-untyped-defs --check-untyped-defs

.PHONY: all install run debug lint lint-strict test clean fclean re

all: install

# Install dependencies with uv.
# On a disk-constrained machine .venv may be a symlink to a bigger partition.
# If its target directory is missing, a bare `uv sync` fails with "File
# exists", so recreate it first. On a fresh clone (moulinette) .venv is not a
# symlink and this step is skipped, leaving a plain `uv sync`.
install:
	@if [ -L .venv ]; then \
		target=$$(readlink .venv); \
		echo "note: .venv -> $$target (ensuring it exists)"; \
		mkdir -p "$$target"; \
	fi
	uv sync

# Run the project. Default paths: data/input/ -> data/output/.
run:
	uv run python -m src $(ARGS)

# Run the project under the standard-library debugger (pdb).
debug:
	uv run python -m pdb -m src $(ARGS)

# flake8 + mypy on our code. llm_sdk (provided) and .venv are excluded via
# .flake8 and pyproject.toml so the subject's `flake8 .` / `mypy .` still work.
lint:
	uv run flake8 .
	uv run mypy . $(MYPY_FLAGS)

lint-strict:
	uv run flake8 .
	uv run mypy . --strict

# Run the test suite. Exit code 5 (pytest found no tests yet) is tolerated.
test:
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
	@if [ -L .venv ]; then rm -rf "$$(readlink .venv)"; fi
	rm -rf .venv
	rm -rf data/output

re: fclean install
