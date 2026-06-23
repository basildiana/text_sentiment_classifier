"""Hyperparameter configuration and result dataclasses for the text sentiment classifier."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import List

import torch


@dataclass
class EvalResult:
    """Holds accuracy, loss, and confusion-matrix counts for one evaluation pass.

    Attributes:
        accuracy: Fraction of correct predictions, in [0, 1].
        loss: Mean BCE loss over the evaluation set.
        tp: True-positive count.
        tn: True-negative count.
        fp: False-positive count.
        fn: False-negative count.
    """

    accuracy: float
    loss: float
    tp: int
    tn: int
    fp: int
    fn: int

    def __post_init__(self) -> None:
        # Clamp accuracy to [0, 1] to satisfy Requirement 8.3.
        self.accuracy = max(0.0, min(1.0, float(self.accuracy)))

    @property
    def precision(self) -> float:
        """Positive predictive value: TP / (TP + FP). Returns 0 when undefined."""
        denom = self.tp + self.fp
        return self.tp / denom if denom > 0 else 0.0

    @property
    def recall(self) -> float:
        """Sensitivity / true-positive rate: TP / (TP + FN). Returns 0 when undefined."""
        denom = self.tp + self.fn
        return self.tp / denom if denom > 0 else 0.0

    @property
    def f1(self) -> float:
        """Harmonic mean of precision and recall. Returns 0 when both are 0."""
        p, r = self.precision, self.recall
        denom = p + r
        return 2 * p * r / denom if denom > 0 else 0.0


@dataclass
class TrainingResult:
    """Aggregates per-epoch loss and evaluation history for a complete training run.

    Attributes:
        train_losses: Mean training loss per epoch (length == config.epochs).
        eval_results: EvalResult for each evaluation checkpoint.
        best_accuracy: Highest accuracy observed across all evaluations.
        best_epoch: Epoch index at which best_accuracy was achieved.
    """

    train_losses: List[float]
    eval_results: List[EvalResult]
    best_accuracy: float
    best_epoch: int


@dataclass
class TrainingConfig:
    """All hyperparameters and I/O paths for a training run.

    Validation rules (enforced in __post_init__):
    - max_len > 0
    - batch_size > 0
    - lr in (0, 1)
    - glove_path must point to a readable file (skipped when value is empty string)
    - model_name must be registered in ModelFactory (checked lazily to avoid circular imports)
    """

    # ── Data ──────────────────────────────────────────────────────────────────
    csv_path: str = ""
    glove_path: str = ""
    max_len: int = 200
    batch_size: int = 32

    # ── Training ──────────────────────────────────────────────────────────────
    epochs: int = 10
    lr: float = 1e-3
    eval_every: int = 1  # evaluate on test set every N epochs

    # ── Model ─────────────────────────────────────────────────────────────────
    model_name: str = "gru"  # one of: rnn, gru, mlp, attention_mlp
    hidden_dim: int = 128
    freeze_embeddings: bool = True
    dropout: float = 0.3
    attn_window: int = 5  # window size for RestrictedAttention

    # ── I/O ───────────────────────────────────────────────────────────────────
    checkpoint_dir: str = "checkpoints/"
    device: str = field(
        default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu"
    )

    def __post_init__(self) -> None:
        """Validate hyperparameters and raise descriptive ValueError on bad input."""
        if self.max_len <= 0:
            raise ValueError(
                f"max_len must be greater than 0, got {self.max_len!r}."
            )
        if self.batch_size <= 0:
            raise ValueError(
                f"batch_size must be greater than 0, got {self.batch_size!r}."
            )
        if not (0.0 < self.lr < 1.0):
            raise ValueError(
                f"lr must be in the open interval (0, 1), got {self.lr!r}."
            )
        # Only validate glove_path when it has been explicitly set.
        if self.glove_path and not os.path.isfile(self.glove_path):
            raise ValueError(
                f"glove_path does not point to a readable file: {self.glove_path!r}."
            )
        # model_name validation is deferred to factory.py to avoid circular imports.
