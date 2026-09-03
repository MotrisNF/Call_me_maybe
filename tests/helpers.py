"""A weight-free, character-level fake model for the unit tests.

The real ``llm_sdk.Small_LLM_Model`` needs ~1.5 GB of weights. For unit tests
we only need something exposing ``encode``, ``decode`` and
``get_logits_from_input_ids`` that behaves deterministically. ``FakeModel``
treats every character of a fixed alphabet as its own token and, when asked for
logits, returns a spike on the next character of a fixed script.
"""

from src.vocab import DIGITS, Vocab


class _Row:
    """One row of a fake ``encode`` result; supports ``.tolist()``."""

    def __init__(self, ids: list[int]) -> None:
        self._ids = ids

    def tolist(self) -> list[int]:
        return list(self._ids)


class _Batch:
    """A fake ``encode`` result; ``[0]`` yields the only :class:`_Row`."""

    def __init__(self, ids: list[int]) -> None:
        self._ids = ids

    def __getitem__(self, index: int) -> _Row:
        return _Row(self._ids)


class FakeModel:
    """Character-level stand-in for ``Small_LLM_Model``.

    Args:
        script: The characters the model should "want" to emit, in order, one
            per :meth:`get_logits_from_input_ids` call.
        alphabet: Every character the model can tokenize; a character's index
            is its token id.
    """

    def __init__(self, script: str, alphabet: str) -> None:
        self._script = script
        self._step = 0
        self._alphabet = alphabet
        self._index = {ch: i for i, ch in enumerate(alphabet)}
        self.size = len(alphabet)

    def encode(self, text: str) -> _Batch:
        return _Batch([self._index[ch] for ch in text])

    def decode(self, ids: list[int]) -> str:
        return "".join(self._alphabet[i] for i in ids)

    def get_logits_from_input_ids(self, input_ids: list[int]) -> list[float]:
        want = self._script[self._step]
        self._step += 1
        logits = [0.0] * self.size
        logits[self._index[want]] = 100.0
        return logits


def make_vocab(alphabet: str) -> Vocab:
    """Build a :class:`Vocab` for a character-level alphabet."""
    id_to_text = list(alphabet)
    return Vocab(
        id_to_text=id_to_text,
        text_to_id={ch: i for i, ch in enumerate(id_to_text)},
        size=len(id_to_text),
        digit_run_ids=frozenset(
            i for i, ch in enumerate(id_to_text) if ch in DIGITS
        ),
    )
