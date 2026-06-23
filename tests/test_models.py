"""Tests for model architectures, ModelFactory, and forward-pass shapes.

**Validates: Requirements 4.1, 4.5, 4.6, 4.7, 4.8, 4.9, 5.1, 5.2, 5.3, 5.4**
"""

from __future__ import annotations

import pytest
import torch
from hypothesis import given, settings
from hypothesis import strategies as st

from text_sentiment_classifier.config import TrainingConfig
from text_sentiment_classifier.factory import ModelFactory, _ensure_models_registered
from text_sentiment_classifier.models.base import BaseClassifier
from text_sentiment_classifier.models.attention_mlp import AttentionMLPClassifier

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

VOCAB_SIZE = 50
EMBED_DIM = 16
SEQ_LEN = 20


@pytest.fixture(scope="module")
def embedding_matrix() -> torch.Tensor:
    torch.manual_seed(0)
    return torch.randn(VOCAB_SIZE, EMBED_DIM)


@pytest.fixture(scope="module")
def config() -> TrainingConfig:
    return TrainingConfig(
        max_len=SEQ_LEN,
        hidden_dim=32,
        dropout=0.0,
        freeze_embeddings=False,
        attn_window=3,
    )


# ---------------------------------------------------------------------------
# Unit tests — ModelFactory
# ---------------------------------------------------------------------------


class TestModelFactory:
    def test_all_four_architectures_registered(self) -> None:
        _ensure_models_registered()
        for name in ("rnn", "gru", "mlp", "attention_mlp"):
            assert name in ModelFactory._registry, f"{name!r} not registered."

    def test_create_returns_base_classifier(
        self, embedding_matrix: torch.Tensor, config: TrainingConfig
    ) -> None:
        _ensure_models_registered()
        for name in ModelFactory._registry:
            model = ModelFactory.create(name, embedding_matrix, config)
            assert isinstance(model, BaseClassifier), (
                f"ModelFactory.create({name!r}) must return a BaseClassifier."
            )

    def test_unknown_name_raises_value_error(
        self, embedding_matrix: torch.Tensor, config: TrainingConfig
    ) -> None:
        with pytest.raises(ValueError, match="Unknown model"):
            ModelFactory.create("nonexistent_model", embedding_matrix, config)

    def test_error_message_includes_valid_names(
        self, embedding_matrix: torch.Tensor, config: TrainingConfig
    ) -> None:
        with pytest.raises(ValueError) as exc_info:
            ModelFactory.create("bad_name", embedding_matrix, config)
        error_msg = str(exc_info.value)
        assert "bad_name" in error_msg
        assert "rnn" in error_msg or "gru" in error_msg


# ---------------------------------------------------------------------------
# Unit tests — forward output shape
# ---------------------------------------------------------------------------


class TestForwardShape:
    @pytest.mark.parametrize("model_name", ["rnn", "gru", "mlp", "attention_mlp"])
    def test_output_shape_batch_1(
        self,
        model_name: str,
        embedding_matrix: torch.Tensor,
        config: TrainingConfig,
    ) -> None:
        model = ModelFactory.create(model_name, embedding_matrix, config)
        model.eval()
        x = torch.zeros(1, SEQ_LEN, dtype=torch.long)
        with torch.no_grad():
            logits = model(x)
        assert logits.shape == (1, 1), (
            f"{model_name}: expected shape (1, 1), got {logits.shape}."
        )

    @pytest.mark.parametrize("model_name", ["rnn", "gru", "mlp", "attention_mlp"])
    def test_output_shape_batch_8(
        self,
        model_name: str,
        embedding_matrix: torch.Tensor,
        config: TrainingConfig,
    ) -> None:
        model = ModelFactory.create(model_name, embedding_matrix, config)
        model.eval()
        x = torch.zeros(8, SEQ_LEN, dtype=torch.long)
        with torch.no_grad():
            logits = model(x)
        assert logits.shape == (8, 1)

    def test_attention_weights_sum_to_one(
        self, embedding_matrix: torch.Tensor, config: TrainingConfig
    ) -> None:
        """Attention weights must sum to 1 along the sequence dimension."""
        model: AttentionMLPClassifier = ModelFactory.create(
            "attention_mlp", embedding_matrix, config
        )  # type: ignore[assignment]
        model.eval()
        B = 4
        x = torch.zeros(B, SEQ_LEN, dtype=torch.long)
        with torch.no_grad():
            model(x)
        weights = model.attention.last_weights  # [B, L]
        assert weights is not None
        sums = weights.sum(dim=-1)              # [B]
        assert torch.allclose(sums, torch.ones(B), atol=1e-5), (
            f"Attention weights do not sum to 1: {sums}"
        )

    def test_predict_proba_in_unit_interval(
        self, embedding_matrix: torch.Tensor, config: TrainingConfig
    ) -> None:
        model = ModelFactory.create("gru", embedding_matrix, config)
        model.eval()
        x = torch.randint(0, VOCAB_SIZE, (4, SEQ_LEN))
        with torch.no_grad():
            probs = model.predict_proba(x)
        assert probs.shape == (4, 1)
        assert torch.all(probs >= 0.0) and torch.all(probs <= 1.0)


# ---------------------------------------------------------------------------
# Property-based test — Property 4: all registered names → valid instances
# ---------------------------------------------------------------------------


@given(st.data())
@settings(max_examples=20)
def test_all_registered_names_produce_valid_instances(data: st.DataObject) -> None:
    """**Property 4: All registered model names produce valid instances**

    **Validates: Requirements 5.1, 5.4**

    For every name in ModelFactory._registry, create() must return an
    instance of BaseClassifier.
    """
    _ensure_models_registered()
    emb = torch.randn(VOCAB_SIZE, EMBED_DIM)
    cfg = TrainingConfig(
        max_len=SEQ_LEN,
        hidden_dim=32,
        dropout=0.0,
        freeze_embeddings=False,
        attn_window=3,
    )
    for name in ModelFactory._registry:
        model = ModelFactory.create(name, emb, cfg)
        assert isinstance(model, BaseClassifier), (
            f"Expected BaseClassifier, got {type(model)} for model {name!r}."
        )


# ---------------------------------------------------------------------------
# Property-based test — Property 3: forward output shape for any batch size
# ---------------------------------------------------------------------------


@given(
    model_name=st.sampled_from(["rnn", "gru", "mlp", "attention_mlp"]),
    batch_size=st.integers(min_value=1, max_value=16),
)
@settings(max_examples=60)
def test_forward_shape_property(model_name: str, batch_size: int) -> None:
    """**Property 3: Model output shape matches batch size**

    **Validates: Requirements 4.1**

    For any batch size in [1, 16] and any registered architecture,
    forward(x).shape must equal (batch_size, 1).
    """
    _ensure_models_registered()
    emb = torch.randn(VOCAB_SIZE, EMBED_DIM)
    cfg = TrainingConfig(
        max_len=SEQ_LEN,
        hidden_dim=32,
        dropout=0.0,
        freeze_embeddings=False,
        attn_window=3,
    )
    model = ModelFactory.create(model_name, emb, cfg)
    model.eval()
    x = torch.zeros(batch_size, SEQ_LEN, dtype=torch.long)
    with torch.no_grad():
        logits = model(x)
    assert logits.shape == (batch_size, 1), (
        f"{model_name} with batch_size={batch_size}: "
        f"expected ({batch_size}, 1), got {logits.shape}."
    )
