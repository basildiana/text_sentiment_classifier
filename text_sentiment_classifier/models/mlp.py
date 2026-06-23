"""MLPClassifier: token-wise MLP with mean-pooling sentiment classifier."""

from __future__ import annotations

import torch
import torch.nn as nn

from text_sentiment_classifier.config import TrainingConfig
from text_sentiment_classifier.models.base import BaseClassifier


class MLPClassifier(BaseClassifier):
    """Binary sentiment classifier using a token-wise multi-layer perceptron.

    Architecture::

        embedding → Linear(E, hidden) → ReLU → Dropout
        → mean-pool over seq_len → Linear(hidden, 1)

    Each token embedding is projected independently to ``hidden_dim``;
    the resulting token representations are averaged over the sequence
    to form a fixed-size document vector.

    Args:
        embedding_matrix: Float32 tensor ``[vocab_size, embed_dim]``.
        config:           Training / model configuration.
    """

    def __init__(
        self, embedding_matrix: torch.Tensor, config: TrainingConfig
    ) -> None:
        super().__init__(embedding_matrix, config)
        embed_dim = embedding_matrix.shape[1]
        self.token_proj = nn.Linear(embed_dim, config.hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(config.dropout)
        self.head = nn.Linear(config.hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the token-wise MLP forward pass.

        Args:
            x: ``LongTensor`` of shape ``[batch, seq_len]``.

        Returns:
            ``FloatTensor`` of shape ``[batch, 1]`` — raw logits.
        """
        embedded = self.embedding(x)                # [B, L, E]
        hidden = self.dropout(self.relu(self.token_proj(embedded)))  # [B, L, H]
        pooled = hidden.mean(dim=1)                  # [B, H]
        return self.head(pooled)                     # [B, 1]


# ---------------------------------------------------------------------------
# Self-registration (triggered by factory.py importing this module).
# ---------------------------------------------------------------------------
from text_sentiment_classifier.factory import ModelFactory  # noqa: E402

ModelFactory._registry["mlp"] = MLPClassifier
