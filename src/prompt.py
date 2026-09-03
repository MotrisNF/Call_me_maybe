"""Build the natural-language prompt sent to the model.

The prompt lists the available functions and asks for a single JSON call.
Constrained decoding does the heavy lifting afterwards, but a clear prompt
still makes the model pick the right function and arguments far more often.
"""

from src.schema import FunctionDef

_SYSTEM = (
    "You convert the user's request into exactly one function call.\n"
    "Available functions:\n"
    "{catalogue}\n"
    "Rules:\n"
    "- Reply with only a JSON object, no prose, no code fences.\n"
    '- Shape: {{"name": "<function>", "parameters": {{"<arg>": <value>}}}}\n'
    "- Copy argument values exactly as they appear in the request; do not "
    "add, translate or reformat them.\n"
    "- For a regex substitution, translate the description into a real "
    'pattern (e.g. "all vowels" -> [aeiou], "asterisks" -> *).\n'
    '- If no function fits the request, use "none" as the name with empty '
    "parameters.\n"
    "Examples:\n"
    "Request: What is the sum of 40 and 2?\n"
    'Answer: {{"name": "fn_add_numbers", "parameters": {{"a": 40, "b": 2}}}}\n'
    "Request: Reverse the string 'abc'\n"
    'Answer: {{"name": "fn_reverse_string", "parameters": {{"s": "abc"}}}}\n'
    "Request: Replace all digits in 'a1b2' with #\n"
    'Answer: {{"name": "fn_substitute_string_with_regex", "parameters": '
    '{{"source_string": "a1b2", "regex": "[0-9]", "replacement": "#"}}}}\n'
    "Request: What is the weather today?\n"
    'Answer: {{"name": "none", "parameters": {{}}}}'
)


def _describe(fn: FunctionDef) -> str:
    """Render one catalogue line: ``- name(a: type, ...): description``."""
    args = ", ".join(f"{key}: {spec.type}"
                     for key, spec in fn.parameters.items())
    return f"- {fn.name}({args}): {fn.description}"


def build_call_prompt(functions: list[FunctionDef], user_prompt: str) -> str:
    """Assemble the chat-formatted prompt for one user request.

    Args:
        functions: The validated function catalogue.
        user_prompt: The natural-language request to translate.

    Returns:
        The full string to tokenize. The trailing empty ``<think></think>``
        block disables Qwen3's reasoning monologue.
    """
    catalogue = "\n".join(_describe(fn) for fn in functions)
    system = _SYSTEM.format(catalogue=catalogue)
    return (
        "<|im_start|>system\n" + system + "<|im_end|>\n"
        "<|im_start|>user\n" + user_prompt + "<|im_end|>\n"
        "<|im_start|>assistant\n<think>\n\n</think>\n\n"
    )
