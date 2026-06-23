"""Evaluation metric helpers: accuracy and confusion-matrix computation."""

from __future__ import annotations

from typing import Tuple

import torch


def compute_accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """Compute binary classification accuracy.

    Applies a sigmoid to logits and thresholds at 0.5 to produce predictions.

    Args:
        logits: Raw model output of shape ``[batch, 1]`` or ``[batch]``.
        labels: Ground-truth labels of shape ``[batch, 1]`` or ``[batch]``.
                Values are expected to be 0 or 1.

    Returns:
        Accuracy as a float in ``[0, 1]``.
    """
    probs = torch.sigmoid(logits.float()).squeeze(-1)
    preds = (probs > 0.5).long()
    targets = labels.long().squeeze(-1)
    correct = (preds == targets).sum().item()
    total = targets.numel()
    return correct / total if total > 0 else 0.0


def compute_confusion(
    logits: torch.Tensor, labels: torch.Tensor
) -> Tuple[int, int, int, int]:
    """Compute confusion-matrix counts for binary classification.

    Args:
        logits: Raw model output of shape ``[batch, 1]`` or ``[batch]``.
        labels: Ground-truth labels of shape ``[batch, 1]`` or ``[batch]``.
                Values are expected to be 0 or 1.

    Returns:
        A tuple ``(tp, tn, fp, fn)`` of non-negative integer counts.
    """
    probs = torch.sigmoid(logits.float()).squeeze(-1)
    preds = (probs > 0.5).long()
    targets = labels.long().squeeze(-1)

    tp = int(((preds == 1) & (targets == 1)).sum().item())
    tn = int(((preds == 0) & (targets == 0)).sum().item())
    fp = int(((preds == 1) & (targets == 0)).sum().item())
    fn = int(((preds == 0) & (targets == 1)).sum().item())
    return tp, tn, fp, fn
