"""Checkpointer: saves the best-accuracy model checkpoint during training."""

from __future__ import annotations

import logging
import os

import torch

from text_sentiment_classifier.config import EvalResult
from text_sentiment_classifier.models.base import BaseClassifier

logger = logging.getLogger(__name__)


class Checkpointer:
    """Persists model weights whenever validation accuracy strictly improves.

    Only the model's ``state_dict`` is saved (not the full model object),
    which keeps checkpoints portable across Python versions.

    Args:
        save_dir:   Directory where checkpoint files are written.
                    Created automatically if it does not exist.
        model_name: Used as part of the checkpoint filename.
    """

    def __init__(self, save_dir: str, model_name: str) -> None:
        self.save_dir = save_dir
        self.model_name = model_name
        self.best_accuracy: float = -1.0
        self._checkpoint_path = os.path.join(
            save_dir, f"{model_name}_best.pt"
        )
        os.makedirs(save_dir, exist_ok=True)

    def maybe_save(
        self,
        epoch: int,
        model: BaseClassifier,
        metrics: EvalResult,
    ) -> bool:
        """Save the model state dict if ``metrics.accuracy`` strictly improves.

        Args:
            epoch:   Current training epoch (used for logging only).
            model:   The model whose weights should be saved.
            metrics: Evaluation metrics for the current checkpoint.

        Returns:
            ``True`` if the checkpoint was saved, ``False`` otherwise.
        """
        if metrics.accuracy > self.best_accuracy:
            self.best_accuracy = metrics.accuracy
            torch.save(model.state_dict(), self._checkpoint_path)
            logger.info(
                "Epoch %d: saved checkpoint with accuracy=%.4f to %s",
                epoch,
                metrics.accuracy,
                self._checkpoint_path,
            )
            return True
        return False

    def load_best(self, model: BaseClassifier) -> BaseClassifier:
        """Restore the weights saved at the highest-accuracy epoch.

        Args:
            model: A model instance whose architecture must match the saved
                   checkpoint.  Weights are loaded in-place.

        Returns:
            The same ``model`` instance with updated weights.

        Raises:
            FileNotFoundError: If no checkpoint has been saved yet.
        """
        if not os.path.isfile(self._checkpoint_path):
            raise FileNotFoundError(
                f"No checkpoint found at {self._checkpoint_path!r}.  "
                "Call maybe_save() at least once before load_best()."
            )
        state_dict = torch.load(self._checkpoint_path, map_location="cpu")
        model.load_state_dict(state_dict)
        logger.info("Loaded best checkpoint from %s", self._checkpoint_path)
        return model
