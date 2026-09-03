"""Tests for :mod:`src.constrained` (grammar helpers and full generation)."""

import pytest

from src.constrained import (
    DecodeError,
    _argmax_over,
    _coerce,
    _gen_choice,
    _gen_number,
    _prefix_ids,
    _span_remainders,
    _string_advance,
    generate_call,
)
from src.schema import load_function_defs
from tests.helpers import FakeModel, make_vocab

_ALPHABET = "abcdefghijklmnopqrstuvwxyz0123456789 {}\":,._-"


def test_string_advance() -> None:
    assert _string_advance(0, '"') == 1          # opening quote
    assert _string_advance(0, "x") == -1         # body before quote
    assert _string_advance(1, "abc") == 1        # plain body
    assert _string_advance(1, '"') == 3          # closing quote
    assert _string_advance(1, "\n") == -1        # raw control char
    assert _string_advance(1, "\\t") == 1        # valid escape
    assert _string_advance(1, "\\x") == -1       # invalid escape
    assert _string_advance(3, ",") == -1         # nothing after the close


def test_span_remainders() -> None:
    assert _span_remainders("greet bob", "")[0] == "greet bob"
    assert _span_remainders("greet bob", "b") == ["ob", ""]   # b twice
    assert _span_remainders("a b a", "a") == [" b a", ""]
    # a span stops at a quote (cannot appear raw in a JSON string)
    assert _span_remainders('say "hi"', "say ") == [""]


def test_prefix_ids() -> None:
    vocab = make_vocab("abc")
    assert _prefix_ids(vocab, ["ab", "ac"]) == {0}          # only "a" a token
    assert _prefix_ids(vocab, ["zzz"]) == set()


def test_argmax_over() -> None:
    assert _argmax_over([0.1, 9.0, 0.2], {0, 1, 2}) == 1
    assert _argmax_over([0.1, 9.0, 0.2], {0, 2}) == 2
    with pytest.raises(DecodeError):
        _argmax_over([0.1], set())


def test_coerce() -> None:
    assert _coerce("42", "integer") == 42
    assert _coerce("4.5", "number") == 4.5
    assert _coerce("true", "boolean") is True
    assert _coerce('"hi"', "string") == "hi"


def test_gen_number_stops_on_terminator() -> None:
    vocab = make_vocab(_ALPHABET)
    model = FakeModel(script="345}", alphabet=_ALPHABET)
    ids: list[int] = []
    stop_id = model.encode("}}")[0].tolist()[0]
    value = _gen_number(model, vocab, ids, "number", "}}", stop_id)
    assert value == "345"


def test_gen_choice_injects_unique_tail() -> None:
    vocab = make_vocab(_ALPHABET)
    model = FakeModel(script="fn_g", alphabet=_ALPHABET)
    ids: list[int] = []
    targets = ['fn_greet"', 'fn_add"', 'none"']
    assert _gen_choice(model, vocab, ids, targets, 64) == 'fn_greet"'
    assert model.decode(ids) == 'fn_greet"'   # tail was injected, not guessed


def test_generate_call_happy_path() -> None:
    vocab = make_vocab(_ALPHABET)
    functions = load_function_defs([
        {"name": "fn_greet", "parameters": {"name": {"type": "string"}}},
        {"name": "fn_add", "parameters": {"a": {"type": "number"}}},
    ])
    model = FakeModel(script='fn_gbob"', alphabet=_ALPHABET)
    call = generate_call(model, vocab, "x", "greet bob", functions)
    assert call == {"name": "fn_greet", "parameters": {"name": "bob"}}


def test_generate_call_none_sentinel() -> None:
    vocab = make_vocab(_ALPHABET)
    functions = load_function_defs([
        {"name": "fn_greet", "parameters": {"name": {"type": "string"}}},
    ])
    model = FakeModel(script="none", alphabet=_ALPHABET)
    call = generate_call(model, vocab, "x", "what is the weather", functions)
    assert call == {"name": "none", "parameters": {}}
