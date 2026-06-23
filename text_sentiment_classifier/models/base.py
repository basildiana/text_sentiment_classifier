"""BaseClassifier: abstract base class for all sentiment classifier architectures."""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
import torch.nn as nn

from text_sentiment_classifier.config import TrainingConfig


class BaseClassifier(nn.Module, ABC):
    """Abstract base that enforces the shared forward contract.

    All concrete classifiers must subclass this and implement ``forward``.
    Pre-trained embeddings are loaded via ``nn.Embedding.from_pretrained``
    and optionally frozen based on ``config.freeze_embeddings``.

    Args:
        embedding_matrix: Float32 tensor of shape ``[vocab_size, embed_dim]``.
        config:           Training / model configuration.
    """

    def __init__(
        self, embedding_matrix: torch.Tensor, config: TrainingConfig
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding.from_pretrained(
            embedding_matrix.float(),
            freeze=config.freeze_embeddings,
            padding_idx=0,
        )

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the classifier forward pass.

        Args:
            x: ``LongTensor`` of shape ``[batch, seq_len]`` containing
               vocabulary indices.

        Returns:
            ``FloatTensor`` of shape ``[batch, 1]`` containing raw logits
            (before sigmoid activation).
        """

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Convenience method: return sigmoid-activated probabilities.

        Args:
            x: Same input as ``forward``.

        Returns:
            ``FloatTensor`` of shape ``[batch, 1]`` with values in ``(0, 1)``.
        """
        return torch.sigmoid(self.forward(x))
