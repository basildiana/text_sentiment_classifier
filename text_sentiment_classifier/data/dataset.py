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
_SPLIT_COL = "split"

# Sentiment string → binary label mapping.
_LABEL_MAP = {"positive": 1, "negative": 0}


class SentimentDataset(Dataset):
    """PyTorch Dataset that loads IMDB reviews from a CSV file.

    The CSV must contain at least three columns:

    - ``review``   — raw review text (string)
    - ``sentiment`` — ``"positive"`` or ``"negative"``
    - ``split``    — ``"train"`` or ``"test"``

    Each call to ``__getitem__`` runs the full ``TextPreprocessor`` pipeline
    and returns a ``(token_ids, label)`` tuple ready for use in a DataLoader.

    Args:
        csv_path:     Path to the IMDB CSV file.
        preprocessor: A fitted :class:`~text_sentiment_classifier.data.preprocessor.TextPreprocessor`.
        split:        Which data split to expose — ``"train"`` or ``"test"``.
    """

    def __init__(
        self,
        csv_path: str,
        preprocessor: TextPreprocessor,
        split: Literal["train", "test"],
    ) -> None:
        df = pd.read_csv(csv_path)

        missing = {_REVIEW_COL, _LABEL_COL, _SPLIT_COL} - set(df.columns)
        if missing:
            raise ValueError(
                f"CSV file is missing required columns: {missing}.  "
                f"Found: {list(df.columns)}"
            )

        df = df[df[_SPLIT_COL] == split].reset_index(drop=True)

        if len(df) == 0:
            raise ValueError(
                f"No rows found for split={split!r} in {csv_path!r}."
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
