# ABOUTME: Punto de entrada de `uv run python -m src`.
# ABOUTME: De momento solo hace bootstrap: carga los JSON de entrada y
# ABOUTME: descarga/comprueba el modelo LLM. La decodificacion restringida
# ABOUTME: se implementara encima de esto.

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
    """Parsea los argumentos de linea de comandos.

    Args:
        argv: Lista de argumentos. Si es ``None`` se usa ``sys.argv``.

    Returns:
        El espacio de nombres con ``functions_definition``, ``input`` y
        ``output`` como objetos :class:`pathlib.Path`.
    """
    parser = argparse.ArgumentParser(
        prog="python -m src",
        description=(
            "Traduce prompts en lenguaje natural a llamadas de funcion "
            "usando decodificacion restringida."
        ),
    )
    parser.add_argument(
        "--functions_definition",
        type=Path,
        default=DEFAULT_FUNCTIONS,
        help="JSON con las funciones disponibles.",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help="JSON con los prompts a procesar.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Fichero JSON de salida.",
    )
    return parser.parse_args(argv)


def load_json(path: Path) -> Any:
    """Carga un fichero JSON gestionando los errores con elegancia.

    Args:
        path: Ruta al fichero JSON.

    Returns:
        El objeto Python resultante de deserializar el JSON.

    Raises:
        SystemExit: Si el fichero no existe o no contiene JSON valido.
    """
    try:
        with path.open(encoding="utf-8") as handle:
            return json.load(handle)
    except FileNotFoundError:
        sys.exit(f"error: no se encuentra el fichero '{path}'")
    except json.JSONDecodeError as exc:
        sys.exit(f"error: '{path}' no es JSON valido: {exc}")
    except OSError as exc:
        sys.exit(f"error: no se puede leer '{path}': {exc}")


def bootstrap_model() -> None:
    """Instancia el modelo para forzar su descarga y comprobar el SDK.

    La primera vez descarga los pesos de ``Qwen/Qwen3-0.6B`` (~1.2 GB) a
    ``~/.cache/huggingface``. Las siguientes llamadas usan la cache local.
    """
    try:
        from llm_sdk import Small_LLM_Model
    except ImportError as exc:
        sys.exit(f"error: no se puede importar llm_sdk: {exc}")

    print(f"[bootstrap] cargando modelo '{MODEL_NAME}' (puede tardar)...")
    model = Small_LLM_Model(model_name=MODEL_NAME)

    # Comprobacion minima de que el SDK responde.
    token_ids = model.encode("What is the sum of 2 and 3?")
    logits = model.get_logits_from_input_ids(token_ids[0].tolist())
    vocab_path = model.get_path_to_vocab_file()

    print(f"[bootstrap] tokens del prompt de prueba: {token_ids.shape[1]}")
    print(f"[bootstrap] tamano del vocabulario (logits): {len(logits)}")
    print(f"[bootstrap] fichero de vocabulario: {vocab_path}")
    print("[bootstrap] modelo listo.")


def main(argv: list[str] | None = None) -> int:
    """Punto de entrada del programa.

    Args:
        argv: Argumentos de linea de comandos (para tests).

    Returns:
        Codigo de salida del proceso (0 si todo va bien).
    """
    args = parse_args(argv)

    functions = load_json(args.functions_definition)
    prompts = load_json(args.input)

    print(f"[bootstrap] funciones cargadas: {len(functions)}")
    print(f"[bootstrap] prompts cargados:  {len(prompts)}")

    bootstrap_model()

    print("[bootstrap] TODO: implementar la decodificacion restringida y")
    print(f"[bootstrap]       escribir el resultado en '{args.output}'.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
