"""AttentionMLPClassifier: token-wise MLP with restricted self-attention pooling."""

from __future__ import annotations

import torch
import torch.nn as nn

from text_sentiment_classifier.config import TrainingConfig
from text_sentiment_classifier.models.base import BaseClassifier
from text_sentiment_classifier.models.layers import RestrictedAttention


class AttentionMLPClassifier(BaseClassifier):
    """Binary sentiment classifier using a token-wise MLP and restricted attention.

    Architecture::

        embedding → token_mlp (Linear → ReLU → Dropout)
        → RestrictedAttention(window=config.attn_window)
        → Linear(hidden, 1)

    Each token is first projected to ``hidden_dim`` by a shared MLP.
    A window-restricted dot-product self-attention then pools the token
    representations into a single context vector, which is fed to the
    classification head.

    Args:
        embedding_matrix: Float32 tensor ``[vocab_size, embed_dim]``.
        config:           Training / model configuration.
                          Must expose ``hidden_dim``, ``dropout``,
                          and ``attn_window``.
    """

    def __init__(
        self, embedding_matrix: torch.Tensor, config: TrainingConfig
    ) -> None:
        super().__init__(embedding_matrix, config)
        embed_dim = embedding_matrix.shape[1]
        self.token_mlp = nn.Sequential(
            nn.Linear(embed_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
        )
        self.attention = RestrictedAttention(
            hidden_dim=config.hidden_dim,
            window=config.attn_window,
        )
        self.head = nn.Linear(config.hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the attention MLP forward pass.

        Args:
            x: ``LongTensor`` of shape ``[batch, seq_len]``.

        Returns:
            ``FloatTensor`` of shape ``[batch, 1]`` — raw logits.

        Postcondition:
            ``self.attention.last_weights.sum(dim=-1)`` is all-ones
            (attention weights sum to 1 along the sequence dimension).
        """
        embedded = self.embedding(x)          # [B, L, E]
        hidden = self.token_mlp(embedded)     # [B, L, H]
        context = self.attention(hidden)      # [B, H]
        return self.head(context)             # [B, 1]


# ---------------------------------------------------------------------------
# Self-registration (triggered by factory.py importing this module).
# ---------------------------------------------------------------------------
from text_sentiment_classifier.factory import ModelFactory  # noqa: E402

ModelFactory._registry["attention_mlp"] = AttentionMLPClassifier
