"""Write the results file."""

import json
from pathlib import Path
from typing import Any


def write_results(path: Path, results: list[dict[str, Any]]) -> None:
    """Serialize ``results`` as a JSON array, creating the parent directory.

    Args:
        path: Destination file; its parent directory is created if missing.
        results: The list of ``{prompt, name, parameters}`` objects.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
