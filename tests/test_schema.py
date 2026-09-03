"""Tests for :mod:`src.schema`."""

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from src.schema import load_function_defs, params_model

_CATALOGUE = json.loads(
    Path("data/input/functions_definition.json").read_text(encoding="utf-8")
)


def test_load_real_catalogue() -> None:
    defs = load_function_defs(_CATALOGUE)
    assert [d.name for d in defs][:2] == ["fn_add_numbers", "fn_greet"]
    assert defs[0].parameters["a"].type == "number"


def test_load_rejects_non_list() -> None:
    with pytest.raises(ValueError):
        load_function_defs({"not": "a list"})


def test_load_rejects_missing_name() -> None:
    with pytest.raises(ValueError):
        load_function_defs([{"description": "no name here"}])


def test_load_rejects_prefix_name_collision() -> None:
    payload = [
        {"name": "fn_a", "parameters": {}},
        {"name": "fn_abc", "parameters": {}},
    ]
    with pytest.raises(ValueError):
        load_function_defs(payload)


def test_params_model_accepts_correct_call() -> None:
    add = load_function_defs(_CATALOGUE)[0]
    model = params_model(add)
    checked = model.model_validate({"a": 2, "b": 3})
    assert checked.model_dump() == {"a": 2.0, "b": 3.0}


def test_params_model_rejects_missing_and_mistyped() -> None:
    add = load_function_defs(_CATALOGUE)[0]
    model = params_model(add)
    with pytest.raises(ValidationError):
        model.model_validate({"a": 2})
    with pytest.raises(ValidationError):
        model.model_validate({"a": "two", "b": 3})
