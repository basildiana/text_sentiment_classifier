"""GloVeLoader: reads a GloVe .txt file and produces a vocabulary + embedding matrix."""

from __future__ import annotations

import logging
from typing import Dict, Tuple

import torch

logger = logging.getLogger(__name__)


class GloVeLoader:
    """Loads pre-trained GloVe word vectors from a text file.

    The GloVe file format is one token per line: the word followed by
    ``dim`` space-separated float values.  Vocabulary index 0 is always
    reserved for the ``<PAD>`` token, whose embedding is the zero vector.

    Example::

        vocab, matrix = GloVeLoader().load("glove.6B.100d.txt", dim=100)
        embedding = nn.Embedding.from_pretrained(matrix, freeze=True)
    """

    def load(
        self, path: str, dim: int = 100
    ) -> Tuple[Dict[str, int], torch.Tensor]:
        """Parse a GloVe file and return a vocabulary and embedding matrix.

        Args:
            path: Path to the GloVe ``.txt`` file.
            dim:  Expected embedding dimension.  Lines with a different
                  number of float values are silently skipped.

        Returns:
            A tuple ``(vocab, embedding_matrix)`` where:

            - ``vocab`` maps each word to its integer index.  Index 0 is
              reserved for ``<PAD>``.
            - ``embedding_matrix`` has shape ``[vocab_size, dim]`` and
              dtype ``torch.float32``.  Row 0 is the zero vector (PAD).

        Raises:
            FileNotFoundError: If ``path`` does not exist or is not readable.
                The error is raised before any memory is allocated.
        """
        import os

        if not os.path.isfile(path):
            raise FileNotFoundError(
                f"GloVe file not found at {path!r}.\n"
                "Download GloVe vectors from https://nlp.stanford.edu/projects/glove/\n"
                "and extract the file (e.g. glove.6B.100d.txt) to the expected path."
            )

        vocab: Dict[str, int] = {"<PAD>": 0}
        vectors = [torch.zeros(dim, dtype=torch.float32)]  # index 0 = PAD

        with open(path, encoding="utf-8") as fh:
            for line_no, line in enumerate(fh, start=1):
                parts = line.rstrip().split(" ")
                if len(parts) < 2:
                    logger.warning("Skipping empty line %d in %s.", line_no, path)
                    continue

                word = parts[0]
                raw_values = parts[1:]

                if len(raw_values) != dim:
                    logger.warning(
                        "Line %d: expected %d floats for word %r, got %d — skipping.",
                        line_no,
                        dim,
                        word,
                        len(raw_values),
                    )
                    continue

                try:
                    vector = torch.tensor(
                        [float(v) for v in raw_values], dtype=torch.float32
                    )
                except ValueError:
                    logger.warning(
                        "Line %d: could not parse vector for word %r — skipping.",
                        line_no,
                        word,
                    )
                    continue

                vocab[word] = len(vocab)
                vectors.append(vector)

        embedding_matrix = torch.stack(vectors)  # [vocab_size, dim]

        assert len(vocab) == embedding_matrix.shape[0], (
            f"Vocab size {len(vocab)} != matrix rows {embedding_matrix.shape[0]}."
        )
        assert embedding_matrix.dtype == torch.float32

        return vocab, embedding_matrix
