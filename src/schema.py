"""Pydantic models for the function catalogue and for validating calls.

Two jobs:

* validate ``functions_definition.json`` when it is loaded
  (:func:`load_function_defs`);
* build, per function, a model that checks a generated ``parameters`` object
  has the right keys and value types (:func:`params_model`).
"""

from typing import Any

from pydantic import BaseModel, ValidationError, create_model

# JSON type name -> Python type used for validation / coercion.
_PY_TYPES: dict[str, type] = {
    "number": float,
    "integer": int,
    "string": str,
    "boolean": bool,
}


class ParamSpec(BaseModel):
    """One parameter's declared type, e.g. ``{"type": "number"}``."""

    type: str


class FunctionDef(BaseModel):
    """A single entry of ``functions_definition.json``."""

    name: str
    description: str = ""
    parameters: dict[str, ParamSpec] = {}
    returns: ParamSpec | None = None


def load_function_defs(raw: Any) -> list[FunctionDef]:
    """Validate the parsed ``functions_definition.json`` payload.

    Args:
        raw: Whatever ``json.load`` produced.

    Returns:
        The list of validated function definitions.

    Raises:
        ValueError: If the payload is not a list, an entry is malformed, or a
            name is a prefix of another (which would make the name grammar in
            :mod:`src.constrained` ambiguous).
    """
    if not isinstance(raw, list):
        raise ValueError("functions_definition.json must contain a JSON array")

    try:
        defs = [FunctionDef.model_validate(item) for item in raw]
    except ValidationError as exc:
        raise ValueError(f"malformed function entry: {exc}") from exc

    names = [d.name for d in defs]
    for short in names:
        for other in names:
            if short != other and other.startswith(short):
                raise ValueError(
                    f"function name '{short}' is a prefix of '{other}'"
                )
    return defs


def params_model(fn: FunctionDef) -> type[BaseModel]:
    """Build a pydantic model that validates a call's ``parameters``.

    Args:
        fn: The chosen function definition.

    Returns:
        A dynamically created ``BaseModel`` subclass with one required field
        per declared parameter, typed from its ``"type"`` (unknown type names
        fall back to ``str``).
    """
    fields: dict[str, Any] = {
        key: (_PY_TYPES.get(spec.type, str), ...)
        for key, spec in fn.parameters.items()
    }
    return create_model(f"{fn.name}_params", **fields)
