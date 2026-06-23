"""Tests for TextPreprocessor.

Includes unit tests for each pipeline step and a property-based test that
verifies the output length invariant across arbitrary string inputs.

**Validates: Requirements 1.1, 1.2, 1.3, 1.4, 1.5**
"""

from __future__ import annotations

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from text_sentiment_classifier.data.preprocessor import TextPreprocessor

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

VOCAB = {
    "<PAD>": 0,
    "great": 1,
    "movie": 2,
    "absolutely": 3,
    "loved": 4,
    "it": 5,
    "the": 6,
    "was": 7,
    "terrible": 8,
}


@pytest.fixture()
def preprocessor() -> TextPreprocessor:
    return TextPreprocessor(max_len=10, vocab=VOCAB)


# ---------------------------------------------------------------------------
# Unit tests — clean()
# ---------------------------------------------------------------------------


class TestClean:
    def test_lowercases_text(self, preprocessor: TextPreprocessor) -> None:
        assert preprocessor.clean("HELLO World") == "hello world"

    def test_removes_digits(self, preprocessor: TextPreprocessor) -> None:
        result = preprocessor.clean("movie123")
        assert "1" not in result and "2" not in result and "3" not in result

    def test_removes_punctuation(self, preprocessor: TextPreprocessor) -> None:
        result = preprocessor.clean("great! movie.")
        assert "!" not in result and "." not in result

    def test_keeps_spaces(self, preprocessor: TextPreprocessor) -> None:
        result = preprocessor.clean("hello world")
        assert " " in result

    def test_empty_string(self, preprocessor: TextPreprocessor) -> None:
        assert preprocessor.clean("") == ""

    def test_only_non_alpha(self, preprocessor: TextPreprocessor) -> None:
        # Numbers and punctuation only → empty or whitespace after cleaning.
        result = preprocessor.clean("123!@#")
        assert result.strip() == ""


# ---------------------------------------------------------------------------
# Unit tests — tokenize()
# ---------------------------------------------------------------------------


class TestTokenize:
    def test_splits_on_whitespace(self, preprocessor: TextPreprocessor) -> None:
        assert preprocessor.tokenize("great movie") == ["great", "movie"]

    def test_empty_string_returns_empty_list(self, preprocessor: TextPreprocessor) -> None:
        assert preprocessor.tokenize("") == []

    def test_multiple_spaces_handled(self, preprocessor: TextPreprocessor) -> None:
        tokens = preprocessor.tokenize("great  movie")
        assert tokens == ["great", "movie"]


# ---------------------------------------------------------------------------
# Unit tests — encode()
# ---------------------------------------------------------------------------


class TestEncode:
    def test_known_tokens_mapped_correctly(self, preprocessor: TextPreprocessor) -> None:
        ids = preprocessor.encode(["great", "movie"])
        assert ids == [1, 2]

    def test_unknown_token_maps_to_zero(self, preprocessor: TextPreprocessor) -> None:
        ids = preprocessor.encode(["unknownword"])
        assert ids == [0]

    def test_empty_list(self, preprocessor: TextPreprocessor) -> None:
        assert preprocessor.encode([]) == []


# ---------------------------------------------------------------------------
# Unit tests — pad()
# ---------------------------------------------------------------------------


class TestPad:
    def test_pads_short_sequence(self, preprocessor: TextPreprocessor) -> None:
        result = preprocessor.pad([1, 2, 3])
        assert result == [1, 2, 3, 0, 0, 0, 0, 0, 0, 0]
        assert len(result) == preprocessor.max_len

    def test_truncates_long_sequence(self, preprocessor: TextPreprocessor) -> None:
        long_ids = list(range(20))
        result = preprocessor.pad(long_ids)
        assert len(result) == preprocessor.max_len
        assert result == long_ids[: preprocessor.max_len]

    def test_exact_length_unchanged(self, preprocessor: TextPreprocessor) -> None:
        ids = list(range(10))
        result = preprocessor.pad(ids)
        assert result == ids

    def test_empty_list_fully_padded(self, preprocessor: TextPreprocessor) -> None:
        result = preprocessor.pad([])
        assert result == [0] * preprocessor.max_len


# ---------------------------------------------------------------------------
# Unit tests — process()
# ---------------------------------------------------------------------------


class TestProcess:
    def test_basic_review(self, preprocessor: TextPreprocessor) -> None:
        result = preprocessor.process("Great movie!")
        assert len(result) == preprocessor.max_len
        assert result[0] == 1  # "great" → index 1

    def test_output_length_on_empty_string(self, preprocessor: TextPreprocessor) -> None:
        result = preprocessor.process("")
        assert len(result) == preprocessor.max_len
        assert all(v == 0 for v in result)

    def test_output_length_on_long_text(self, preprocessor: TextPreprocessor) -> None:
        long_text = "great " * 50
        result = preprocessor.process(long_text)
        assert len(result) == preprocessor.max_len


# ---------------------------------------------------------------------------
# Property-based test — Property 1: output length is always exactly max_len
# ---------------------------------------------------------------------------


@given(
    max_len=st.integers(min_value=1, max_value=100),
    text=st.text(),
)
@settings(max_examples=300)
def test_process_output_length_property(max_len: int, text: str) -> None:
    """**Property 1: Preprocessor output length is always exactly max_len**

    **Validates: Requirements 1.1, 1.4, 1.5**

    For any combination of max_len > 0 and any string input, process()
    must return a list of exactly max_len integers.
    """
    preprocessor = TextPreprocessor(max_len=max_len, vocab=VOCAB)
    result = preprocessor.process(text)
    assert len(result) == max_len, (
        f"Expected length {max_len}, got {len(result)} for text {text!r}"
    )
