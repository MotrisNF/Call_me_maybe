"""Tests for :mod:`src.output` and :mod:`src.prompt`."""

import json
from pathlib import Path

from src.output import write_results
from src.prompt import build_call_prompt
from src.schema import load_function_defs

_CATALOGUE = json.loads(
    Path("data/input/functions_definition.json").read_text(encoding="utf-8")
)


def test_write_results_creates_dir_and_valid_json(tmp_path: Path) -> None:
    target = tmp_path / "nested" / "results.json"
    results = [
        {"prompt": "hi", "name": "fn_greet", "parameters": {"name": "x"}},
    ]
    write_results(target, results)

    assert target.exists()
    assert json.loads(target.read_text(encoding="utf-8")) == results


def test_build_call_prompt_mentions_functions_and_request() -> None:
    functions = load_function_defs(_CATALOGUE)
    prompt = build_call_prompt(functions, "Greet bob")

    assert "fn_add_numbers" in prompt
    assert "fn_substitute_string_with_regex" in prompt
    assert "Greet bob" in prompt
    assert "<|im_start|>assistant" in prompt
    assert prompt.endswith("</think>\n\n")
