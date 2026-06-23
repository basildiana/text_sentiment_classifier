"""Custom neural-network layers: BilinearLayer and RestrictedAttention."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class BilinearLayer(nn.Module):
    """Learned bilinear transformation: output = x @ W @ y^T.

    Useful for flexible tensor contraction in attention variants.

    Args:
        in_features:  Dimension of both input tensors.
        out_features: Output dimension.
    """

    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(in_features, out_features))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Compute the bilinear form between x and y.

        Args:
            x: Tensor with last dimension == in_features.
            y: Tensor with last dimension == in_features.

        Returns:
            Tensor of shape ``(*batch, out_features)``.
        """
        # x @ W  → (..., out_features)
        xw = x @ self.weight  # (..., out_features)
        # contract with y: (..., out_features) * (..., out_features) element-wise
        # or broadcast via matmul depending on caller's intent.
        return xw @ y.transpose(-2, -1)


class RestrictedAttention(nn.Module):
    """Window-masked dot-product self-attention with softmax normalisation.

    Each token can only attend to the ``window`` tokens on either side of it
    (plus itself).  Positions outside the window receive a ``-inf`` mask
    before softmax, so they contribute zero weight.

    The weighted-mean context vector is returned, and the raw attention
    weights are stored in ``self.last_weights`` for inspection / testing.

    Args:
        hidden_dim: Dimensionality of the input token representations.
        window:     Half-width of the attention window (tokens attended =
                    ``2 * window + 1`` at most).
    """

    def __init__(self, hidden_dim: int, window: int) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.window = window
        # Scale factor for dot-product attention (Vaswani et al., 2017).
        self._scale = math.sqrt(hidden_dim)
        # Populated during forward() for testability.
        self.last_weights: torch.Tensor | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply restricted self-attention and return a weighted context vector.

        Args:
            x: Input tensor of shape ``[batch, seq_len, hidden_dim]``.

        Returns:
            Context tensor of shape ``[batch, hidden_dim]`` — the
            attention-weighted mean over the sequence dimension.

        Postcondition:
            ``self.last_weights.sum(dim=-1)`` is all-ones (weights sum to 1).
        """
        B, L, H = x.shape  # batch, seq_len, hidden_dim

        # Scaled dot-product scores: [B, L, L]
        scores = torch.bmm(x, x.transpose(1, 2)) / self._scale

        # Build window mask: positions more than `window` steps away are masked.
        # Shape: [L, L]; True where attention is *forbidden*.
        positions = torch.arange(L, device=x.device)
        distance = (positions.unsqueeze(0) - positions.unsqueeze(1)).abs()  # [L, L]
        mask = distance > self.window  # [L, L]

        # Apply mask: set forbidden positions to -inf before softmax.
        scores = scores.masked_fill(mask.unsqueeze(0), float("-inf"))

        # Softmax over the key dimension (last dim = seq_len).
        weights = F.softmax(scores, dim=-1)  # [B, L, L]

        # Weighted sum over value tokens: [B, L, H]
        attended = torch.bmm(weights, x)  # [B, L, H]

        # Pool to a single context vector by averaging over token positions.
        # First average over L (query positions) to get [B, H].
        context = attended.mean(dim=1)  # [B, H]

        # Store per-token attention weights for the *first* query position
        # (representative; averaged over queries for testing convenience).
        # Shape: [B, L] — sum along dim=-1 should be 1 per row.
        self.last_weights = weights.mean(dim=1)  # [B, L]

        return context
