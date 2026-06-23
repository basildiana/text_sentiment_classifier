"""GRUClassifier: single-layer GRU sentiment classifier."""

from __future__ import annotations

import torch
import torch.nn as nn

from text_sentiment_classifier.config import TrainingConfig
from text_sentiment_classifier.models.base import BaseClassifier


class GRUClassifier(BaseClassifier):
    """Binary sentiment classifier using a single-layer ``nn.GRU``.

    Architecture::

        embedding → GRU(batch_first=True) → dropout → linear head

    The final hidden state of the GRU is used as the sequence representation.

    Args:
        embedding_matrix: Float32 tensor ``[vocab_size, embed_dim]``.
        config:           Training / model configuration.
    """

    def __init__(
        self, embedding_matrix: torch.Tensor, config: TrainingConfig
    ) -> None:
        super().__init__(embedding_matrix, config)
        embed_dim = embedding_matrix.shape[1]
        self.gru = nn.GRU(
            input_size=embed_dim,
            hidden_size=config.hidden_dim,
            batch_first=True,
        )
        self.dropout = nn.Dropout(config.dropout)
        self.head = nn.Linear(config.hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the GRU forward pass.

        Args:
            x: ``LongTensor`` of shape ``[batch, seq_len]``.

        Returns:
            ``FloatTensor`` of shape ``[batch, 1]`` — raw logits.
        """
        embedded = self.embedding(x)             # [B, L, E]
        _, hidden = self.gru(embedded)           # hidden: [1, B, H]
        hidden = self.dropout(hidden.squeeze(0)) # [B, H]
        return self.head(hidden)                 # [B, 1]


# ---------------------------------------------------------------------------
# Self-registration (triggered by factory.py importing this module).
# ---------------------------------------------------------------------------
from text_sentiment_classifier.factory import ModelFactory  # noqa: E402

ModelFactory._registry["gru"] = GRUClassifier
