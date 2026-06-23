"""TextPreprocessor: converts raw review strings into padded integer token sequences."""

from __future__ import annotations

import re
from typing import Dict, List


class TextPreprocessor:
    """Normalises text and encodes it as a fixed-length integer sequence.

    The pipeline is: clean → tokenize → encode → pad.  Unknown tokens are
    mapped to index 0 (the PAD/UNK slot), matching the GloVe convention.

    Args:
        max_len: Fixed output length.  Longer sequences are truncated;
                 shorter ones are zero-padded.
        vocab: Mapping from word string to integer index.  Index 0 is
               reserved for ``<PAD>`` / unknown tokens.
    """

    def __init__(self, max_len: int, vocab: Dict[str, int]) -> None:
        if max_len <= 0:
            raise ValueError(f"max_len must be > 0, got {max_len!r}.")
        self.max_len = max_len
        self.vocab = vocab

    # ------------------------------------------------------------------
    # Individual pipeline steps
    # ------------------------------------------------------------------

    def clean(self, text: str) -> str:
        """Lowercase the text and remove every character that is not a-z or whitespace.

        Args:
            text: Raw input string.

        Returns:
            A lowercase string containing only alphabetic characters and whitespace.
        """
        return re.sub(r"[^a-zA-Z\s]", "", text.lower())

    def tokenize(self, text: str) -> List[str]:
        """Split a cleaned string on whitespace.

        Args:
            text: A pre-cleaned (or arbitrary) string.

        Returns:
            A list of word tokens.  Empty strings after split are excluded.
        """
        return text.split()

    def encode(self, tokens: List[str]) -> List[int]:
        """Map each token to its vocabulary index; unknown tokens map to 0.

        Args:
            tokens: List of word strings.

        Returns:
            List of integer vocabulary indices.
        """
        return [self.vocab.get(tok, 0) for tok in tokens]

    def pad(self, ids: List[int]) -> List[int]:
        """Truncate or zero-pad ``ids`` to exactly ``self.max_len`` elements.

        Args:
            ids: List of integer token ids.

        Returns:
            A list of exactly ``self.max_len`` integers.
        """
        ids = ids[: self.max_len]
        ids = ids + [0] * (self.max_len - len(ids))
        return ids

    # ------------------------------------------------------------------
    # Public composite entry-point
    # ------------------------------------------------------------------

    def process(self, text: str) -> List[int]:
        """Run the full clean → tokenize → encode → pad pipeline.

        Args:
            text: A raw review string (may be empty or contain any characters).

        Returns:
            A list of exactly ``self.max_len`` integer vocabulary indices.

        Postcondition:
            ``len(result) == self.max_len`` for any input string.
        """
        cleaned = self.clean(text)
        tokens = self.tokenize(cleaned)
        ids = self.encode(tokens)
        result = self.pad(ids)
        assert len(result) == self.max_len, (
            f"Expected {self.max_len} tokens, got {len(result)}."
        )
        return result
