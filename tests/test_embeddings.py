"""Tests for GloVeLoader.

Includes unit tests for normal loading behaviour and error paths, plus a
property-based test verifying the PAD invariant and shape consistency across
synthetically generated GloVe files.

**Validates: Requirements 2.1, 2.2, 2.3, 2.4, 2.5, 11.1, 11.2**
"""

from __future__ import annotations

import os
import tempfile
from typing import Generator

import pytest
import torch
from hypothesis import given, settings
from hypothesis import strategies as st

from text_sentiment_classifier.data.embeddings import GloVeLoader


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def write_glove_file(path: str, words: list[str], dim: int) -> None:
    """Write a minimal GloVe-format text file."""
    import random
    with open(path, "w", encoding="utf-8") as fh:
        for word in words:
            values = " ".join(f"{random.uniform(-1, 1):.6f}" for _ in range(dim))
            fh.write(f"{word} {values}\n")


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------


class TestGloVeLoaderUnit:
    def test_pad_index_is_zero(self, tmp_path) -> None:
        glove_path = str(tmp_path / "glove.txt")
        write_glove_file(glove_path, ["hello", "world"], dim=10)
        vocab, emb = GloVeLoader().load(glove_path, dim=10)
        assert vocab["<PAD>"] == 0

    def test_pad_vector_is_zero(self, tmp_path) -> None:
        glove_path = str(tmp_path / "glove.txt")
        write_glove_file(glove_path, ["hello", "world"], dim=10)
        _, emb = GloVeLoader().load(glove_path, dim=10)
        assert torch.all(emb[0] == 0.0)

    def test_vocab_and_matrix_same_size(self, tmp_path) -> None:
        glove_path = str(tmp_path / "glove.txt")
        words = ["foo", "bar", "baz"]
        write_glove_file(glove_path, words, dim=5)
        vocab, emb = GloVeLoader().load(glove_path, dim=5)
        assert len(vocab) == emb.shape[0]

    def test_matrix_shape(self, tmp_path) -> None:
        glove_path = str(tmp_path / "glove.txt")
        words = ["a", "b", "c"]
        dim = 8
        write_glove_file(glove_path, words, dim=dim)
        vocab, emb = GloVeLoader().load(glove_path, dim=dim)
        # vocab_size = 1 (<PAD>) + len(words) = 4
        assert emb.shape == (len(words) + 1, dim)

    def test_dtype_is_float32(self, tmp_path) -> None:
        glove_path = str(tmp_path / "glove.txt")
        write_glove_file(glove_path, ["test"], dim=4)
        _, emb = GloVeLoader().load(glove_path, dim=4)
        assert emb.dtype == torch.float32

    def test_missing_file_raises_file_not_found(self) -> None:
        with pytest.raises(FileNotFoundError):
            GloVeLoader().load("/nonexistent/path/glove.txt", dim=100)

    def test_wrong_dim_line_skipped(self, tmp_path) -> None:
        """Lines with wrong vector length should be silently skipped."""
        path = str(tmp_path / "mixed.txt")
        with open(path, "w") as fh:
            fh.write("good 0.1 0.2 0.3\n")        # 3 floats, matches dim=3
            fh.write("bad  0.1 0.2\n")             # 2 floats, should skip
        vocab, emb = GloVeLoader().load(path, dim=3)
        assert "good" in vocab
        assert "bad" not in vocab

    def test_words_assigned_sequential_indices(self, tmp_path) -> None:
        glove_path = str(tmp_path / "glove.txt")
        words = ["alpha", "beta", "gamma"]
        write_glove_file(glove_path, words, dim=4)
        vocab, _ = GloVeLoader().load(glove_path, dim=4)
        # Index 0 is PAD; words start from 1.
        for expected_idx, word in enumerate(words, start=1):
            assert vocab[word] == expected_idx


# ---------------------------------------------------------------------------
# Property-based test — Property 2: PAD invariant and shape consistency
# ---------------------------------------------------------------------------


@given(
    words=st.lists(
        st.text(
            alphabet=st.characters(whitelist_categories=("Ll",)),
            min_size=1,
            max_size=12,
        ),
        min_size=1,
        max_size=30,
        unique=True,
    ),
    dim=st.integers(min_value=2, max_value=16),
)
@settings(max_examples=100)
def test_glove_pad_invariant_property(words: list[str], dim: int) -> None:
    """**Property 2: GloVe PAD token is always index 0 with a zero vector**

    **Validates: Requirements 2.2, 11.1, 11.2**

    For any valid synthetic GloVe file:
    - vocab["<PAD>"] == 0
    - emb[0] is the zero vector
    - len(vocab) == emb.shape[0]
    - emb.dtype == torch.float32
    """
    import random
    import tempfile

    with tempfile.TemporaryDirectory() as tmp_dir:
        glove_path = os.path.join(tmp_dir, "glove.txt")
        with open(glove_path, "w", encoding="utf-8") as fh:
            for word in words:
                values = " ".join(f"{random.uniform(-1, 1):.6f}" for _ in range(dim))
                fh.write(f"{word} {values}\n")

        vocab, emb = GloVeLoader().load(glove_path, dim=dim)

    assert vocab["<PAD>"] == 0, "PAD token must always be at index 0."
    assert torch.all(emb[0] == 0.0), "PAD embedding must be the zero vector."
    assert len(vocab) == emb.shape[0], (
        f"Vocab size {len(vocab)} must equal matrix rows {emb.shape[0]}."
    )
    assert emb.dtype == torch.float32, (
        f"Embedding dtype must be float32, got {emb.dtype}."
    )
