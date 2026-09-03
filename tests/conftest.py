"""Pytest fixtures."""

from collections.abc import Iterator

import pytest

from src import constrained


@pytest.fixture(autouse=True)
def clear_string_cache() -> Iterator[None]:
    """Reset the vocabulary-scoped cache between tests (each uses its own)."""
    constrained._STRING_LEGAL.clear()
    yield
    constrained._STRING_LEGAL.clear()
