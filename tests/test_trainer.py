"""Tests for Trainer — training loop and evaluation.

Includes property-based tests for:
- Property 6: training loss is non-negative
- Property 7: attention weights sum to 1

**Validates: Requirements 4.9, 6.1, 6.2, 6.3, 6.4, 6.5, 6.6, 6.7, 8.1**
"""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader, TensorDataset
from hypothesis import given, settings
from hypothesis import strategies as st

from text_sentiment_classifier.config import TrainingConfig
from text_sentiment_classifier.factory import ModelFactory, _ensure_models_registered
from text_sentiment_classifier.training.trainer import Trainer
from text_sentiment_classifier.models.attention_mlp import AttentionMLPClassifier

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

VOCAB_SIZE = 30
EMBED_DIM = 8
SEQ_LEN = 10
NUM_SAMPLES = 40


def make_synthetic_loader(
    n: int = NUM_SAMPLES,
    seq_len: int = SEQ_LEN,
    batch_size: int = 8,
) -> DataLoader:
    """Build a DataLoader with random integer inputs and binary labels."""
    x = torch.randint(0, VOCAB_SIZE, (n, seq_len))
    y = torch.randint(0, 2, (n, 1))
    ds = TensorDataset(x, y)
    return DataLoader(ds, batch_size=batch_size, shuffle=False)


def make_config(**kwargs) -> TrainingConfig:
    defaults = dict(
        max_len=SEQ_LEN,
        hidden_dim=16,
        dropout=0.0,
        freeze_embeddings=False,
        attn_window=3,
        epochs=1,
        batch_size=8,
        lr=1e-3,
        eval_every=1,
        checkpoint_dir="",  # will be overridden by tmp_path in tests
    )
    defaults.update(kwargs)
    return TrainingConfig(**defaults)


def make_model(model_name: str, config: TrainingConfig):
    _ensure_models_registered()
    emb = torch.randn(VOCAB_SIZE, EMBED_DIM)
    return ModelFactory.create(model_name, emb, config)


# ---------------------------------------------------------------------------
# Unit tests — Trainer
# ---------------------------------------------------------------------------


class TestTrainerUnit:
    def test_train_losses_length_equals_epochs(self, tmp_path) -> None:
        config = make_config(epochs=3, checkpoint_dir=str(tmp_path))
        model = make_model("gru", config)
        trainer = Trainer(config, device=torch.device("cpu"))
        loader = make_synthetic_loader()
        result = trainer.fit(model, loader, loader)
        assert len(result.train_losses) == 3

    def test_best_accuracy_equals_max_eval_accuracy(self, tmp_path) -> None:
        config = make_config(epochs=2, eval_every=1, checkpoint_dir=str(tmp_path))
        model = make_model("mlp", config)
        trainer = Trainer(config, device=torch.device("cpu"))
        loader = make_synthetic_loader()
        result = trainer.fit(model, loader, loader)
        max_acc = max(r.accuracy for r in result.eval_results)
        assert abs(result.best_accuracy - max_acc) < 1e-6

    def test_eval_results_non_empty(self, tmp_path) -> None:
        config = make_config(epochs=2, eval_every=1, checkpoint_dir=str(tmp_path))
        model = make_model("rnn", config)
        trainer = Trainer(config, device=torch.device("cpu"))
        loader = make_synthetic_loader()
        result = trainer.fit(model, loader, loader)
        assert len(result.eval_results) > 0

    def test_evaluate_returns_eval_result(self, tmp_path) -> None:
        from text_sentiment_classifier.config import EvalResult
        config = make_config(epochs=1, checkpoint_dir=str(tmp_path))
        model = make_model("gru", config)
        trainer = Trainer(config, device=torch.device("cpu"))
        loader = make_synthetic_loader()
        result = trainer.evaluate(model, loader)
        assert isinstance(result, EvalResult)
        assert 0.0 <= result.accuracy <= 1.0

    def test_attention_weights_sum_to_one_after_forward(self, tmp_path) -> None:
        """Property 7: attention weights sum to 1 along sequence dimension."""
        config = make_config(epochs=1, checkpoint_dir=str(tmp_path))
        emb = torch.randn(VOCAB_SIZE, EMBED_DIM)
        model: AttentionMLPClassifier = ModelFactory.create(
            "attention_mlp", emb, config
        )  # type: ignore[assignment]
        model.eval()
        B = 4
        x = torch.zeros(B, SEQ_LEN, dtype=torch.long)
        with torch.no_grad():
            model(x)
        weights = model.attention.last_weights  # [B, L]
        assert weights is not None
        sums = weights.sum(dim=-1)
        assert torch.allclose(sums, torch.ones(B), atol=1e-5), (
            f"Attention weights don't sum to 1: {sums}"
        )


# ---------------------------------------------------------------------------
# Property-based test — Property 6: training loss is non-negative
# ---------------------------------------------------------------------------


@given(
    model_name=st.sampled_from(["rnn", "gru", "mlp", "attention_mlp"]),
    n_samples=st.integers(min_value=8, max_value=32),
)
@settings(max_examples=20)
def test_training_loss_non_negative_property(
    model_name: str, n_samples: int,
) -> None:
    """**Property 6: Training loss is non-negative for all epochs**

    **Validates: Requirements 6.7**

    For any architecture and any small synthetic dataset, every training epoch
    must produce a non-negative mean loss.
    """
    import tempfile

    _ensure_models_registered()
    with tempfile.TemporaryDirectory() as tmp_dir:
        config = make_config(
            model_name=model_name,
            epochs=1,
            batch_size=min(8, n_samples),
            checkpoint_dir=tmp_dir,
        )
        model = make_model(model_name, config)
        loader = make_synthetic_loader(n=n_samples, batch_size=min(8, n_samples))
        trainer = Trainer(config, device=torch.device("cpu"))
        result = trainer.fit(model, loader, loader)

    for epoch_idx, loss in enumerate(result.train_losses):
        assert loss >= 0.0, (
            f"Epoch {epoch_idx}: expected non-negative loss, got {loss}."
        )


# ---------------------------------------------------------------------------
# Property-based test — Property 7: attention weights sum to 1
# ---------------------------------------------------------------------------


@given(batch_size=st.integers(min_value=1, max_value=8))
@settings(max_examples=30)
def test_attention_weights_sum_to_one_property(batch_size: int) -> None:
    """**Property 7: Attention weights sum to 1 along sequence dimension**

    **Validates: Requirements 4.9**

    For any batch size, after a forward pass through AttentionMLPClassifier,
    the stored attention weights must sum to 1 along the sequence dimension.
    """
    _ensure_models_registered()
    config = make_config(epochs=1, attn_window=3)
    emb = torch.randn(VOCAB_SIZE, EMBED_DIM)
    model: AttentionMLPClassifier = ModelFactory.create(
        "attention_mlp", emb, config
    )  # type: ignore[assignment]
    model.eval()
    x = torch.zeros(batch_size, SEQ_LEN, dtype=torch.long)
    with torch.no_grad():
        model(x)
    weights = model.attention.last_weights  # [B, L]
    assert weights is not None
    sums = weights.sum(dim=-1)  # [B]
    assert torch.allclose(sums, torch.ones(batch_size), atol=1e-5), (
        f"Attention weights don't sum to 1 for batch_size={batch_size}: {sums}"
    )
