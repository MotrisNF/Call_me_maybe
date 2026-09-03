"""Constrained decoding: force the model's output to be a schema-valid call.

Same loop as :mod:`src.llm` (logits -> pick a token -> append -> repeat) with
one addition: before every pick we keep only the tokens that cannot break the
target JSON.

The output shape is fixed by the subject::

    {"name": "<catalogue name>", "parameters": {"<key>": <value>, ...}}

so rather than a general JSON grammar we walk that shape directly. The fixed
parts are *injected* as tokens (no model call, nothing to choose); only the
holes -- the function name and each argument value -- are generated, each under
its own constraint.

Nothing here uses private ``llm_sdk`` attributes: token text comes from
:class:`src.vocab.Vocab`, built with the public ``decode``.
"""

import json
from typing import Any

from src.schema import FunctionDef
from src.vocab import DIGITS, Vocab

MAX_NAME_CHARS = 64
MAX_VALUE_CHARS = 256
# A string argument is usually a literal span copied from the request, so we
# first try to constrain it to a substring of the prompt. These bound the two
# generation modes when the model will not stop on its own.
MAX_SPAN_CHARS = 160
MAX_FREE_STRING_CHARS = 32

# Cache of "legal token ids" per string sub-state (0, 1, 2). Filled lazily;
# purely a function of the vocabulary, so it never needs clearing.
_STRING_LEGAL: dict[int, frozenset[int]] = {}


class DecodeError(RuntimeError):
    """Raised when the constraint leaves no legal token (a dead end)."""


def _encode(model: Any, text: str) -> list[int]:
    """Tokenize a fixed literal into a flat list of ids."""
    ids: list[int] = model.encode(text)[0].tolist()
    return ids


def _argmax_over(logits: list[float], ids: set[int]) -> int:
    """Greedy pick restricted to ``ids``."""
    if not ids:
        raise DecodeError("no legal token at this position")
    return max(ids, key=logits.__getitem__)


def _prefix_ids(vocab: Vocab, remainders: list[str]) -> set[int]:
    """Ids of every token that is a non-empty prefix of some remainder."""
    out: set[int] = set()
    for rem in remainders:
        for size in range(1, len(rem) + 1):
            token_id = vocab.text_to_id.get(rem[:size])
            if token_id is not None:
                out.add(token_id)
    return out


def _span_remainders(request: str, body: str) -> list[str]:
    """Text that follows each occurrence of ``body`` in ``request``.

    ``body`` is always a substring of ``request`` here, so the only legal next
    tokens are prefixes of one of these remainders. This keeps the per-step
    work proportional to the (short) request instead of the whole vocabulary.
    """
    out: list[str] = []
    start = 0
    while True:
        hit = request.find(body, start)
        if hit < 0:
            return out
        tail = request[hit + len(body):]
        # A JSON string body cannot contain a raw quote, backslash or control
        # character, so a span cannot continue past one.
        cut = len(tail)
        for pos, ch in enumerate(tail):
            if ch in '"\\' or ord(ch) < 0x20:
                cut = pos
                break
        out.append(tail[:cut])
        start = hit + 1


def _gen_choice(model: Any, vocab: Vocab, ids: list[int],
                targets: list[str], cap: int) -> str:
    """Generate the text of exactly one string from ``targets``.

    Args:
        model: The loaded model.
        vocab: Vocabulary tables.
        ids: The running id list; extended in place with the chosen tokens.
        targets: The complete strings the output is allowed to become.
        cap: Safety bound on the generated length.

    Returns:
        The target string that was produced.

    Raises:
        DecodeError: If nothing legal can be picked, or ``cap`` is hit.
    """
    cur = ""
    while len(cur) <= cap:
        matches = [t for t in targets if t.startswith(cur)]
        if not matches:
            raise DecodeError(f"no target matches {cur!r}")
        if cur in matches:
            return cur
        if len(matches) == 1:
            # Only one target is still possible: inject the rest, no more
            # model calls needed.
            ids.extend(_encode(model, matches[0][len(cur):]))
            return matches[0]
        legal = _prefix_ids(vocab, [t[len(cur):] for t in matches])
        logits = model.get_logits_from_input_ids(ids)
        pick = _argmax_over(logits, legal)
        ids.append(pick)
        cur += vocab.id_to_text[pick]
    raise DecodeError(f"value did not terminate (targets={targets!r})")


def _gen_number(model: Any, vocab: Vocab, ids: list[int], vtype: str,
                terminator: str, stop_id: int) -> str:
    """Generate a JSON number; stop when the model emits ``terminator``."""
    minus_id = vocab.text_to_id.get("-")
    dot_id = vocab.text_to_id.get(".")
    value = ""
    while len(value) <= MAX_VALUE_CHARS:
        legal: set[int] = set(vocab.digit_run_ids)
        if value == "" and minus_id is not None:
            legal.add(minus_id)
        has_digit = any(ch in DIGITS for ch in value)
        if (vtype == "number" and dot_id is not None
                and "." not in value and has_digit):
            legal.add(dot_id)

        choices = set(legal)
        can_stop = has_digit and not value.endswith(".")
        if can_stop:
            choices.add(stop_id)

        logits = model.get_logits_from_input_ids(ids)
        pick = _argmax_over(logits, choices)
        if pick == stop_id and pick not in legal:
            ids.extend(_encode(model, terminator))
            return value
        ids.append(pick)
        value += vocab.id_to_text[pick]
    raise DecodeError("number value too long")


def _string_advance(state: int, text: str) -> int:
    """Fold ``text`` through the JSON-string state machine.

    States: 0 = before the opening quote, 1 = in the body, 2 = right after a
    backslash, 3 = closing quote just seen. Returns the new state, or -1 if
    ``text`` is not a legal continuation from ``state``.
    """
    cur = state
    for ch in text:
        if cur == 0:
            cur = 1 if ch == '"' else -1
        elif cur == 1:
            if ch == "\\":
                cur = 2
            elif ch == '"':
                cur = 3
            elif ord(ch) < 0x20:
                cur = -1
            else:
                cur = 1
        elif cur == 2:
            cur = 1 if ch in '"\\/bfnrtu' else -1
        else:  # cur == 3: nothing may follow the closing quote in one token
            cur = -1
        if cur == -1:
            return -1
    return cur


def _string_legal(vocab: Vocab, state: int) -> frozenset[int]:
    """Legal token ids from a string sub-state (cached)."""
    if state not in _STRING_LEGAL:
        _STRING_LEGAL[state] = frozenset(
            i
            for i, text in enumerate(vocab.id_to_text)
            if text != "" and _string_advance(state, text) != -1
        )
    return _STRING_LEGAL[state]


def _gen_string_span(model: Any, vocab: Vocab, ids: list[int], request: str,
                     terminator: str) -> str:
    """Generate a string whose body is a contiguous substring of ``request``.

    The opening quote must already be in ``ids``. This is the common case for
    function arguments: the value is a literal span lifted from the request,
    so the model physically cannot hallucinate one.

    Raises:
        DecodeError: If no substring can be started (the caller then falls
            back to :func:`_gen_string_free`).
    """
    quote_id = vocab.text_to_id.get('"')
    body = ""
    while len(body) <= MAX_SPAN_CHARS:
        legal = _prefix_ids(vocab, _span_remainders(request, body))
        choices = set(legal)
        if body and quote_id is not None:
            choices.add(quote_id)
        logits = model.get_logits_from_input_ids(ids)

        # Escape hatch: if, at the very first character, the model would much
        # rather emit something that is *not* a span of the request (e.g. "["
        # to start a regex), give up and let the caller use free generation.
        if body == "":
            free_best = _argmax_over(logits, set(_string_legal(vocab, 1)))
            if free_best not in choices and (
                logits[free_best] > logits[_argmax_over(logits, choices)]
            ):
                raise DecodeError("model prefers a non-span value")

        pick = _argmax_over(logits, choices)
        if pick == quote_id and pick not in legal:
            ids.append(quote_id)
            ids.extend(_encode(model, terminator))
            return f'"{body}"'
        ids.append(pick)
        body += vocab.id_to_text[pick]

    if quote_id is not None:
        ids.append(quote_id)
    ids.extend(_encode(model, terminator))
    return f'"{body}"'


def _gen_string_free(model: Any, vocab: Vocab, ids: list[int],
                     terminator: str) -> str:
    """Generate a JSON string body with structure-only constraints.

    Used when the value is not a literal span (e.g. a regex the model has to
    infer). The opening quote must already be in ``ids``; generation starts in
    the string body (state 1).
    """
    body = ""
    state = 1
    while len(body) <= MAX_FREE_STRING_CHARS:
        legal = set(_string_legal(vocab, state))
        logits = model.get_logits_from_input_ids(ids)
        pick = _argmax_over(logits, legal)
        text = vocab.id_to_text[pick]
        ids.append(pick)
        body += text
        state = _string_advance(state, text)
        if state == 3:
            ids.extend(_encode(model, terminator))
            return f'"{body}'

    close_id = vocab.text_to_id.get('"')
    if close_id is not None:
        ids.append(close_id)
    ids.extend(_encode(model, terminator))
    return f'"{body}"'


def _gen_value(model: Any, vocab: Vocab, ids: list[int], vtype: str,
               terminator: str, request: str) -> str:
    """Generate one value of ``vtype`` and consume ``terminator``."""
    stop_id = _encode(model, terminator)[0]
    if vtype in ("number", "integer"):
        return _gen_number(model, vocab, ids, vtype, terminator, stop_id)
    if vtype == "boolean":
        text = _gen_choice(model, vocab, ids, ["true", "false"], cap=8)
        ids.extend(_encode(model, terminator))
        return text

    opening = _encode(model, '"')
    mark = len(ids)
    ids.extend(opening)
    try:
        return _gen_string_span(model, vocab, ids, request, terminator)
    except DecodeError:
        del ids[mark + len(opening):]
        return _gen_string_free(model, vocab, ids, terminator)


def _coerce(raw: str, vtype: str) -> Any:
    """Turn the generated text into the Python value of ``vtype``."""
    if vtype == "integer":
        return int(raw)
    if vtype == "number":
        return float(raw)
    if vtype == "boolean":
        return raw == "true"
    return json.loads(raw)  # string: raw still carries its quotes


def generate_call(model: Any, vocab: Vocab, prompt: str, request: str,
                  functions: list[FunctionDef]) -> dict[str, Any]:
    """Generate one schema-valid ``{"name", "parameters"}`` object.

    Args:
        model: The loaded model.
        vocab: Vocabulary lookup tables.
        prompt: The chat-formatted prompt for this request.
        request: The raw user request (used to constrain string values to
            spans copied from it).
        functions: The validated catalogue.

    Returns:
        ``{"name": str, "parameters": dict}`` -- parseable by construction and
        matching one catalogue entry.

    Raises:
        DecodeError: If the constraint dead-ends.
    """
    by_name = {fn.name: fn for fn in functions}
    names = [f'{fn.name}"' for fn in functions]

    ids = _encode(model, prompt)
    ids.extend(_encode(model, '{"name": "'))
    chosen = _gen_choice(model, vocab, ids, names, MAX_NAME_CHARS)[:-1]
    fn = by_name[chosen]

    ids.extend(_encode(model, ', "parameters": {'))
    params: dict[str, Any] = {}
    items = list(fn.parameters.items())
    for index, (key, spec) in enumerate(items):
        if index > 0:
            ids.extend(_encode(model, ", "))
        ids.extend(_encode(model, f'"{key}": '))
        terminator = "}}" if index == len(items) - 1 else ", "
        raw = _gen_value(model, vocab, ids, spec.type, terminator, request)
        params[key] = _coerce(raw, spec.type)
    if not items:
        ids.extend(_encode(model, "}}"))

    return {"name": chosen, "parameters": params}
