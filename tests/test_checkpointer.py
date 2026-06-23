"""Tests for Checkpointer.

Includes unit tests for save/load behaviour and a property-based test that
verifies the monotonicity of save decisions across arbitrary accuracy sequences.

**Validates: Requirements 7.1, 7.2, 7.3**
"""

from __future__ import annotations

import os

import pytest
import torch
from hypothesis import given, settings
from hypothesis import strategies as st

from text_sentiment_classifier.config import EvalResult, TrainingConfig
from text_sentiment_classifier.factory import ModelFactory, _ensure_models_registered
from text_sentiment_classifier.training.checkpointer import Checkpointer


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def make_eval_result(accuracy: float) -> EvalResult:
    return EvalResult(accuracy=accuracy, loss=1.0, tp=0, tn=0, fp=0, fn=0)


def make_model(tmp_dir) -> object:
    """Create a tiny GRU model for checkpoint tests."""
    _ensure_models_registered()
    emb = torch.randn(20, 8)
    cfg = TrainingConfig(
        max_len=5,
        hidden_dim=16,
        dropout=0.0,
        freeze_embeddings=False,
    )
    return ModelFactory.create("gru", emb, cfg)


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------


class TestCheckpointerUnit:
    def test_first_save_returns_true(self, tmp_path) -> None:
        model = make_model(tmp_path)
        ckpt = Checkpointer(str(tmp_path), "gru")
        result = ckpt.maybe_save(0, model, make_eval_result(0.8))
        assert result is True

    def test_checkpoint_file_created(self, tmp_path) -> None:
        model = make_model(tmp_path)
        ckpt = Checkpointer(str(tmp_path), "gru")
        ckpt.maybe_save(0, model, make_eval_result(0.75))
        assert os.path.isfile(os.path.join(str(tmp_path), "gru_best.pt"))

    def test_lower_accuracy_returns_false(self, tmp_path) -> None:
        model = make_model(tmp_path)
        ckpt = Checkpointer(str(tmp_path), "gru")
        ckpt.maybe_save(0, model, make_eval_result(0.9))
        result = ckpt.maybe_save(1, model, make_eval_result(0.7))
        assert result is False

    def test_equal_accuracy_returns_false(self, tmp_path) -> None:
        model = make_model(tmp_path)
        ckpt = Checkpointer(str(tmp_path), "gru")
        ckpt.maybe_save(0, model, make_eval_result(0.8))
        result = ckpt.maybe_save(1, model, make_eval_result(0.8))
        assert result is False

    def test_strictly_higher_accuracy_returns_true(self, tmp_path) -> None:
        model = make_model(tmp_path)
        ckpt = Checkpointer(str(tmp_path), "gru")
        ckpt.maybe_save(0, model, make_eval_result(0.7))
        result = ckpt.maybe_save(1, model, make_eval_result(0.8))
        assert result is True

    def test_load_best_restores_weights(self, tmp_path) -> None:
        model = make_model(tmp_path)
        ckpt = Checkpointer(str(tmp_path), "gru")
        ckpt.maybe_save(0, model, make_eval_result(0.85))

        # Corrupt model weights, then reload.
        with torch.no_grad():
            for p in model.parameters():
                p.fill_(999.0)

        loaded_model = ckpt.load_best(model)
        # After loading, parameters should no longer all be 999.
        any_not_corrupted = any(
            not torch.all(p == 999.0) for p in loaded_model.parameters()
        )
        assert any_not_corrupted

    def test_load_best_without_save_raises(self, tmp_path) -> None:
        model = make_model(tmp_path)
        ckpt = Checkpointer(str(tmp_path / "empty"), "gru")
        with pytest.raises(FileNotFoundError):
            ckpt.load_best(model)

    def test_save_dir_created_automatically(self, tmp_path) -> None:
        new_dir = str(tmp_path / "nested" / "checkpoints")
        model = make_model(tmp_path)
        ckpt = Checkpointer(new_dir, "rnn")
        ckpt.maybe_save(0, model, make_eval_result(0.6))
        assert os.path.isdir(new_dir)


# ---------------------------------------------------------------------------
# Property-based test — Property 5: monotone save decisions
# ---------------------------------------------------------------------------


@given(
    accuracies=st.lists(
        st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
        min_size=2,
        max_size=20,
    )
)
@settings(max_examples=200)
def test_checkpointer_monotone_property(accuracies: list[float]) -> None:
    """**Property 5: Checkpointer only saves when accuracy strictly improves**

    **Validates: Requirements 7.1, 7.2**

    For any sequence of accuracy values, the checkpointer must save if and
    only if the current accuracy strictly exceeds all previously seen values.
    The save decisions are therefore non-decreasing in the running maximum:
    once a save is skipped for value v, no value ≤ v ever causes a save.
    """
    import tempfile

    with tempfile.TemporaryDirectory() as tmp_dir:
        model = make_model(tmp_dir)
        ckpt = Checkpointer(tmp_dir, "test_model")

        running_max = float("-inf")
        for epoch, acc in enumerate(accuracies):
            result = ckpt.maybe_save(epoch, model, make_eval_result(acc))
            if acc > running_max:
                assert result is True, (
                    f"Expected save=True at epoch {epoch} "
                    f"(acc={acc:.4f} > running_max={running_max:.4f})."
                )
                running_max = acc
            else:
                assert result is False, (
                    f"Expected save=False at epoch {epoch} "
                    f"(acc={acc:.4f} <= running_max={running_max:.4f})."
                )
