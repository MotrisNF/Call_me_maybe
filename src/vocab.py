"""Token/text lookup tables derived from the model's own tokenizer.

Constrained decoding must know, for every possible next token, the exact text
that token appends to the output. We read that from the SDK's ``decode`` (one
id at a time) and cache a few tables. Rebuilding this from ``vocab.json`` +
``merges.txt`` is a bonus objective and is deliberately not done here.
"""

from typing import Any

DIGITS = frozenset("0123456789")


class Vocab:
    """Lookup tables over the model vocabulary.

    Attributes:
        id_to_text: ``id_to_text[i]`` is the string token ``i`` decodes to.
            Special/added tokens decode to ``""``.
        text_to_id: reverse map; on the rare duplicate the first id wins.
        size: number of tokens, i.e. the length of one logits vector.
        digit_run_ids: ids whose text is one or more decimal digits.
    """

    def __init__(self, id_to_text: list[str]) -> None:
        self.id_to_text = id_to_text
        self.size = len(id_to_text)

        self.text_to_id: dict[str, int] = {}
        for i, text in enumerate(id_to_text):
            if text and text not in self.text_to_id:
                self.text_to_id[text] = i

        self.digit_run_ids: frozenset[int] = frozenset(
            i
            for i, text in enumerate(id_to_text)
            if text != "" and all(ch in DIGITS for ch in text)
        )


def build_vocab(model: Any) -> Vocab:
    """Decode every token id once and return the populated tables.

    Args:
        model: A loaded ``llm_sdk.Small_LLM_Model``.

    Returns:
        The :class:`Vocab` for this model.
    """
    # The logits vector length is the authoritative vocabulary size (it
    # includes special tokens that are absent from vocab.json).
    probe_ids: list[int] = model.encode("a")[0].tolist()
    size = len(model.get_logits_from_input_ids(probe_ids))

    id_to_text = [str(model.decode([i])) for i in range(size)]
    return Vocab(id_to_text)
