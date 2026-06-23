"""Trainer: owns the training loop, evaluation, and orchestration."""

from __future__ import annotations

import logging
from typing import Optional

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader

from text_sentiment_classifier.config import EvalResult, TrainingConfig, TrainingResult
from text_sentiment_classifier.models.base import BaseClassifier
from text_sentiment_classifier.training.checkpointer import Checkpointer
from text_sentiment_classifier.utils.metrics import compute_accuracy, compute_confusion

logger = logging.getLogger(__name__)


class Trainer:
    """Runs the complete training loop with BCE loss and Adam optimiser.

    Args:
        config: Hyperparameter and I/O configuration.
        device: Torch device on which all tensors and the model reside.
    """

    def __init__(self, config: TrainingConfig, device: torch.device) -> None:
        self.config = config
        self.device = device

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def fit(
        self,
        model: BaseClassifier,
        train_loader: DataLoader,
        test_loader: DataLoader,
        checkpointer: Optional[Checkpointer] = None,
    ) -> TrainingResult:
        """Train ``model`` for ``config.epochs`` epochs.

        Args:
            model:        The classifier to train.  Must already be on
                          ``self.device``.
            train_loader: DataLoader yielding ``(token_ids, labels)`` pairs.
            test_loader:  DataLoader used for periodic evaluation.
            checkpointer: If ``None``, a :class:`Checkpointer` is created
                          automatically from ``config.checkpoint_dir`` and
                          ``config.model_name``.

        Returns:
            A :class:`TrainingResult` with per-epoch losses and evaluation
            history.

        Raises:
            torch.cuda.OutOfMemoryError: Re-raised after logging a helpful
                suggestion to reduce ``batch_size``.
        """
        if checkpointer is None:
            checkpointer = Checkpointer(
                save_dir=self.config.checkpoint_dir,
                model_name=self.config.model_name,
            )

        optimizer = Adam(model.parameters(), lr=self.config.lr)
        criterion = nn.BCEWithLogitsLoss()

        train_losses: list[float] = []
        eval_results: list[EvalResult] = []
        best_accuracy: float = 0.0
        best_epoch: int = 0

        for epoch in range(self.config.epochs):
            model.train()
            epoch_loss = 0.0
            num_batches = 0

            for batch_ids, batch_labels in train_loader:
                batch_ids = batch_ids.to(self.device)
                # Labels are LongTensor [B, 1]; BCE expects FloatTensor.
                batch_labels = batch_labels.float().to(self.device)

                try:
                    optimizer.zero_grad()
                    logits = model(batch_ids)                   # [B, 1]
                    loss = criterion(logits, batch_labels)
                    loss.backward()
                    optimizer.step()
                    epoch_loss += loss.item()
                    num_batches += 1

                except torch.cuda.OutOfMemoryError:
                    logger.error(
                        "CUDA out-of-memory error at epoch %d.  "
                        "Try reducing batch_size (current: %d).",
                        epoch,
                        self.config.batch_size,
                    )
                    raise

            mean_loss = epoch_loss / max(num_batches, 1)
            train_losses.append(mean_loss)
            logger.info("Epoch %d/%d — train loss: %.4f", epoch + 1, self.config.epochs, mean_loss)

            if epoch % self.config.eval_every == 0:
                result = self.evaluate(model, test_loader)
                eval_results.append(result)
                checkpointer.maybe_save(epoch, model, result)
                logger.info(
                    "Epoch %d/%d — eval accuracy: %.4f  loss: %.4f",
                    epoch + 1,
                    self.config.epochs,
                    result.accuracy,
                    result.loss,
                )
                if result.accuracy > best_accuracy:
                    best_accuracy = result.accuracy
                    best_epoch = epoch

        return TrainingResult(
            train_losses=train_losses,
            eval_results=eval_results,
            best_accuracy=best_accuracy,
            best_epoch=best_epoch,
        )

    def evaluate(
        self, model: BaseClassifier, loader: DataLoader
    ) -> EvalResult:
        """Run a full evaluation pass over ``loader``.

        Args:
            model:  The classifier (will be set to eval mode).
            loader: DataLoader yielding ``(token_ids, labels)`` pairs.

        Returns:
            An :class:`EvalResult` with accuracy, loss, and confusion counts
            computed over the entire dataset.
        """
        criterion = nn.BCEWithLogitsLoss()
        model.eval()

        total_loss = 0.0
        num_batches = 0
        tp_total = tn_total = fp_total = fn_total = 0

        with torch.no_grad():
            for batch_ids, batch_labels in loader:
                batch_ids = batch_ids.to(self.device)
                batch_labels_float = batch_labels.float().to(self.device)

                logits = model(batch_ids)  # [B, 1]
                loss = criterion(logits, batch_labels_float)
                total_loss += loss.item()
                num_batches += 1

                tp, tn, fp, fn = compute_confusion(logits.cpu(), batch_labels.cpu())
                tp_total += tp
                tn_total += tn
                fp_total += fp
                fn_total += fn

        mean_loss = total_loss / max(num_batches, 1)
        total_samples = tp_total + tn_total + fp_total + fn_total
        accuracy = (tp_total + tn_total) / total_samples if total_samples > 0 else 0.0

        return EvalResult(
            accuracy=accuracy,
            loss=mean_loss,
            tp=tp_total,
            tn=tn_total,
            fp=fp_total,
            fn=fn_total,
        )
