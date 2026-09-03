"""Run constrained decoding over every prompt and collect the results."""

import sys
from typing import Any

from pydantic import ValidationError

from src.constrained import DecodeError, generate_call
from src.prompt import build_call_prompt
from src.schema import FunctionDef, params_model
from src.vocab import Vocab


def process_prompts(model: Any, vocab: Vocab, functions: list[FunctionDef],
                    prompts: Any) -> list[dict[str, Any]]:
    """Resolve each prompt entry to a function call.

    A prompt that cannot be resolved (missing ``"prompt"`` string, constraint
    dead-end, failed validation) is skipped with a warning on stderr, so the
    output file only ever holds schema-valid ``{prompt, name, parameters}``
    objects.

    Args:
        model: The loaded model.
        vocab: Vocabulary lookup tables.
        functions: Validated function catalogue.
        prompts: The parsed ``function_calling_tests.json`` payload.

    Returns:
        One result object per successfully resolved prompt.
    """
    if not isinstance(prompts, list):
        raise ValueError("the input file must contain a JSON array")

    by_name = {fn.name: fn for fn in functions}
    results: list[dict[str, Any]] = []

    for entry in prompts:
        text = entry.get("prompt") if isinstance(entry, dict) else None
        if not isinstance(text, str):
            print(f"warning: skipping entry without a 'prompt' string: "
                  f"{entry!r}", file=sys.stderr)
            continue
        try:
            call = generate_call(
                model, vocab, build_call_prompt(functions, text), text,
                functions,
            )
            checked = params_model(by_name[call["name"]]).model_validate(
                call["parameters"]
            )
            results.append({
                "prompt": text,
                "name": call["name"],
                "parameters": checked.model_dump(),
            })
        except (DecodeError, ValueError, ValidationError) as exc:
            print(f"warning: skipping prompt {text!r}: {exc}", file=sys.stderr)

    return results
