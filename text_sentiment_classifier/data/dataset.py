"""SentimentDataset: PyTorch Dataset wrapping the IMDB CSV file."""

from __future__ import annotations

from typing import Literal, Tuple

import pandas as pd
import torch
from torch.utils.data import Dataset

from text_sentiment_classifier.data.preprocessor import TextPreprocessor

# Column names expected in the CSV file.
_REVIEW_COL = "review"
_LABEL_COL = "sentiment"

# Sentiment string → binary label mapping.
_LABEL_MAP = {"positive": 1, "negative": 0}

# Fraction of data used for training (the rest becomes the test split).
_TRAIN_RATIO = 0.8


class SentimentDataset(Dataset):
    """PyTorch Dataset that loads IMDB reviews from a CSV file.

    The CSV must contain exactly two columns:

    - ``review``    — raw review text (string)
    - ``sentiment`` — ``"positive"`` or ``"negative"``

    The dataset is split into train / test by row index using an 80/20
    ratio.  No ``split`` column is required.

    Each call to ``__getitem__`` runs the full ``TextPreprocessor`` pipeline
    and returns a ``(token_ids, label)`` tuple ready for use in a DataLoader.

    Args:
        csv_path:     Path to the IMDB CSV file.
        preprocessor: A fitted :class:`~text_sentiment_classifier.data.preprocessor.TextPreprocessor`.
        split:        Which data split to expose — ``"train"`` or ``"test"``.
        train_ratio:  Fraction of rows assigned to the train split (default 0.8).
    """

    def __init__(
        self,
        csv_path: str,
        preprocessor: TextPreprocessor,
        split: Literal["train", "test"],
        train_ratio: float = _TRAIN_RATIO,
    ) -> None:
        df = pd.read_csv(csv_path)

        missing = {_REVIEW_COL, _LABEL_COL} - set(df.columns)
        if missing:
            raise ValueError(
                f"CSV file is missing required columns: {missing}.  "
                f"Found: {list(df.columns)}"
            )

        # Deterministic train/test split by index — no shuffling so results
        # are reproducible without needing a split column in the CSV.
        cutoff = int(len(df) * train_ratio)
        df = df.iloc[:cutoff] if split == "train" else df.iloc[cutoff:]
        df = df.reset_index(drop=True)

        if len(df) == 0:
            raise ValueError(
                f"No rows found for split={split!r} in {csv_path!r}.  "
                f"Check that the file has more than one row."
            )

        self._preprocessor = preprocessor
        self._reviews: list[str] = df[_REVIEW_COL].tolist()
        self._labels: list[int] = [
            _LABEL_MAP.get(str(s).strip().lower(), 0)
            for s in df[_LABEL_COL].tolist()
        ]

    def __len__(self) -> int:
        return len(self._reviews)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return a single preprocessed sample.

        Args:
            idx: Sample index.

        Returns:
            A tuple ``(token_ids, label)`` where:

            - ``token_ids``: ``LongTensor`` of shape ``[max_len]``
            - ``label``:     ``LongTensor`` of shape ``[1]``
        """
        ids = self._preprocessor.process(self._reviews[idx])
        token_ids = torch.tensor(ids, dtype=torch.long)
        label = torch.tensor([self._labels[idx]], dtype=torch.long)
        return token_ids, label
