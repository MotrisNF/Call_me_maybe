"""Minimal (unconstrained) text-generation loop on top of ``llm_sdk``.

This is the plain version of the pipeline from the subject::

    prompt -> tokens -> input ids -> LLM -> logits -> pick next token -> repeat

Nothing here filters the logits, so the model is free to say anything. Later,
constrained decoding will hook into :func:`generate` by editing the logits
*before* :func:`argmax` picks a token.

The model is passed in already built. This module only needs three methods of
``llm_sdk.Small_LLM_Model`` -- ``encode``, ``decode`` and
``get_logits_from_input_ids`` -- so it is typed loosely as ``Any`` and can be
driven by a fake object in tests.
"""

from typing import Any

# Qwen3 is instruction-tuned: it was trained to answer *after* the
# "<|im_start|>assistant" marker. Raw text still generates something, but the
# answers are noticeably worse without this wrapper.
#
# The trailing empty "<think></think>" block disables Qwen3's reasoning mode
# (otherwise it emits a long "<think> ... </think>" monologue before the
# answer). For function calling we want the structured answer directly.
_CHAT_TEMPLATE = (
    "<|im_start|>user\n"
    "{message}<|im_end|>\n"
    "<|im_start|>assistant\n"
    "<think>\n\n</think>\n\n"
)


def build_chat_prompt(message: str) -> str:
    """Wrap a user message in Qwen's chat markers.

    Args:
        message: The natural-language request.

    Returns:
        The full string to tokenize.
    """
    return _CHAT_TEMPLATE.format(message=message)


def argmax(values: list[float]) -> int:
    """Return the index of the largest value (the greedy token choice).

    Args:
        values: The logits, one score per vocabulary token.

    Returns:
        The index -- i.e. the token id -- of the highest logit.
    """
    return max(range(len(values)), key=values.__getitem__)


def generate(model: Any, prompt: str, max_new_tokens: int = 64) -> str:
    """Generate text greedily, one token at a time.

    The loop is: ask the model for the next-token logits given everything so
    far, pick the most likely token, append it, repeat.

    Args:
        model: A loaded ``llm_sdk.Small_LLM_Model`` (or any object exposing
            ``encode``, ``decode`` and ``get_logits_from_input_ids``).
        prompt: The already-formatted prompt string.
        max_new_tokens: Hard cap on how many tokens to generate.

    Returns:
        The decoded text of the newly generated tokens only (the prompt is
        stripped off).
    """
    # encode() returns a 2-D tensor of shape [1, n]; row 0 is the id list.
    input_ids: list[int] = model.encode(prompt)[0].tolist()
    prompt_len = len(input_ids)

    for _ in range(max_new_tokens):
        logits = model.get_logits_from_input_ids(input_ids)
        next_id = argmax(logits)
        input_ids.append(next_id)

        # Stop heuristic that avoids private tokenizer attributes: decode()
        # drops special tokens, so an end marker (<|im_end|>, <|endoftext|>)
        # decodes to "". A hard cap (max_new_tokens) is still the real bound.
        if model.decode([next_id]) == "":
            input_ids.pop()
            break

    text: str = model.decode(input_ids[prompt_len:])
    return text
