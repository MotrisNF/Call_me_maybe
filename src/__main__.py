# ABOUTME: Entry point for `uv run python -m src`.
# ABOUTME: For now it only bootstraps: it loads the input JSON files and
# ABOUTME: downloads/checks the LLM. Constrained decoding will be built on
# ABOUTME: top of this.

import argparse
import json
import sys
from pathlib import Path
from typing import Any

DEFAULT_FUNCTIONS = Path("data/input/functions_definition.json")
DEFAULT_INPUT = Path("data/input/function_calling_tests.json")
DEFAULT_OUTPUT = Path("data/output/function_calling_results.json")

MODEL_NAME = "Qwen/Qwen3-0.6B"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the command-line arguments.

    Args:
        argv: List of arguments. If ``None``, ``sys.argv`` is used.

    Returns:
        The namespace with ``functions_definition``, ``input`` and
        ``output`` as :class:`pathlib.Path` objects.
    """
    parser = argparse.ArgumentParser(
        prog="python -m src",
        description=(
            "Translate natural-language prompts into structured function "
            "calls using constrained decoding."
        ),
    )
    parser.add_argument(
        "--functions_definition",
        type=Path,
        default=DEFAULT_FUNCTIONS,
        help="JSON file with the available functions.",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help="JSON file with the prompts to process.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Output JSON file.",
    )
    return parser.parse_args(argv)


def load_json(path: Path) -> Any:
    """Load a JSON file, handling errors gracefully.

    Args:
        path: Path to the JSON file.

    Returns:
        The Python object produced by deserializing the JSON.

    Raises:
        SystemExit: If the file does not exist or does not contain valid
            JSON.
    """
    try:
        with path.open(encoding="utf-8") as handle:
            return json.load(handle)
    except FileNotFoundError:
        sys.exit(f"error: file '{path}' not found")
    except json.JSONDecodeError as exc:
        sys.exit(f"error: '{path}' is not valid JSON: {exc}")
    except OSError as exc:
        sys.exit(f"error: cannot read '{path}': {exc}")


def bootstrap_model() -> None:
    """Instantiate the model to force its download and check the SDK.

    The first run downloads the ``Qwen/Qwen3-0.6B`` weights (~1.2 GB) into
    ``~/.cache/huggingface``. Later runs use the local cache.
    """
    try:
        from llm_sdk import Small_LLM_Model
    except ImportError as exc:
        sys.exit(f"error: cannot import llm_sdk: {exc}")

    print(f"[bootstrap] loading model '{MODEL_NAME}' (this may take a while)...")
    model = Small_LLM_Model(model_name=MODEL_NAME)

    # Minimal check that the SDK responds.
    token_ids = model.encode("What is the sum of 2 and 3?")
    logits = model.get_logits_from_input_ids(token_ids[0].tolist())
    vocab_path = model.get_path_to_vocab_file()

    print(f"[bootstrap] tokens in the sample prompt: {token_ids.shape[1]}")
    print(f"[bootstrap] vocabulary size (logits): {len(logits)}")
    print(f"[bootstrap] vocabulary file: {vocab_path}")
    print("[bootstrap] model ready.")


def main(argv: list[str] | None = None) -> int:
    """Program entry point.

    Args:
        argv: Command-line arguments (useful for tests).

    Returns:
        Process exit code (0 on success).
    """
    args = parse_args(argv)

    functions = load_json(args.functions_definition)
    prompts = load_json(args.input)

    print(f"[bootstrap] functions loaded: {len(functions)}")
    print(f"[bootstrap] prompts loaded:   {len(prompts)}")

    bootstrap_model()

    print("[bootstrap] TODO: implement constrained decoding and write the")
    print(f"[bootstrap]       result to '{args.output}'.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
