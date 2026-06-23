"""RNNClassifier: single-layer vanilla RNN sentiment classifier."""

from __future__ import annotations

import torch
import torch.nn as nn

from text_sentiment_classifier.config import TrainingConfig
from text_sentiment_classifier.models.base import BaseClassifier


class RNNClassifier(BaseClassifier):
    """Binary sentiment classifier using a single-layer ``nn.RNN``.

    Architecture::

        embedding → RNN(batch_first=True) → dropout → linear head

    The final hidden state of the RNN is used as the sequence representation.

    Args:
        embedding_matrix: Float32 tensor ``[vocab_size, embed_dim]``.
        config:           Training / model configuration.
    """

    def __init__(
        self, embedding_matrix: torch.Tensor, config: TrainingConfig
    ) -> None:
        super().__init__(embedding_matrix, config)
        embed_dim = embedding_matrix.shape[1]
        self.rnn = nn.RNN(
            input_size=embed_dim,
            hidden_size=config.hidden_dim,
            batch_first=True,
        )
        self.dropout = nn.Dropout(config.dropout)
        self.head = nn.Linear(config.hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the RNN forward pass.

        Args:
            x: ``LongTensor`` of shape ``[batch, seq_len]``.

        Returns:
            ``FloatTensor`` of shape ``[batch, 1]`` — raw logits.
        """
        embedded = self.embedding(x)             # [B, L, E]
        _, hidden = self.rnn(embedded)           # hidden: [1, B, H]
        hidden = self.dropout(hidden.squeeze(0)) # [B, H]
        return self.head(hidden)                 # [B, 1]


# ---------------------------------------------------------------------------
# Self-registration: importing this module from factory.py will call register.
# We import ModelFactory here; since factory.py defers model imports inside a
# function (_ensure_models_registered), there is no circular import at load time.
# ---------------------------------------------------------------------------
from text_sentiment_classifier.factory import ModelFactory  # noqa: E402

ModelFactory._registry["rnn"] = RNNClassifier
