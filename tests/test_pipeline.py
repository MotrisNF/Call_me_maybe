"""Tests for :mod:`src.pipeline`."""

import string

from src.pipeline import process_prompts
from src.schema import load_function_defs
from tests.helpers import FakeModel, make_vocab

_ALPHABET = string.printable
_FUNCTIONS = load_function_defs(
    [{"name": "fn_greet", "parameters": {"name": {"type": "string"}}}]
)


def test_skips_entry_without_prompt_string() -> None:
    model = FakeModel(script="", alphabet=_ALPHABET)
    out = process_prompts(model, make_vocab(_ALPHABET), _FUNCTIONS,
                          [{"nope": 1}, "not a dict"])
    assert out == []


def test_skips_prompt_with_no_matching_function() -> None:
    # "n" is enough for the name grammar to lock onto the "none" sentinel;
    # the rest is injected, so a one-character script is all it takes.
    model = FakeModel(script="n", alphabet=_ALPHABET)
    out = process_prompts(model, make_vocab(_ALPHABET), _FUNCTIONS,
                          [{"prompt": "what is the weather today"}])
    assert out == []


def test_non_list_input_raises() -> None:
    model = FakeModel(script="", alphabet=_ALPHABET)
    try:
        process_prompts(model, make_vocab(_ALPHABET), _FUNCTIONS, {"x": 1})
    except ValueError:
        return
    raise AssertionError("expected ValueError for non-list input")
