"""CLI entrypoint for training a text sentiment classifier.

Usage example::

    python train.py \\
        --csv_path data/imdb.csv \\
        --glove_path glove.6B.100d.txt \\
        --model_name gru \\
        --epochs 10 \\
        --batch_size 32

Run ``python train.py --help`` for the full list of options.
"""

from __future__ import annotations

import argparse
import logging
import sys

import torch
from torch.utils.data import DataLoader

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    """Construct the argument parser with all TrainingConfig fields as flags."""
    parser = argparse.ArgumentParser(
        description="Train a binary sentiment classifier on IMDB reviews.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── Data ──────────────────────────────────────────────────────────────
    parser.add_argument("--csv_path", required=True, help="Path to IMDB CSV file.")
    parser.add_argument(
        "--glove_path", required=True, help="Path to GloVe embeddings (.txt)."
    )
    parser.add_argument("--max_len", type=int, default=200, help="Sequence length cap.")
    parser.add_argument("--batch_size", type=int, default=32, help="Mini-batch size.")

    # ── Training ──────────────────────────────────────────────────────────
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Adam learning rate.")
    parser.add_argument(
        "--eval_every",
        type=int,
        default=1,
        help="Evaluate on test set every N epochs.",
    )

    # ── Model ─────────────────────────────────────────────────────────────
    parser.add_argument(
        "--model_name",
        default="gru",
        help="Architecture name: rnn | gru | mlp | attention_mlp.",
    )
    parser.add_argument("--hidden_dim", type=int, default=128, help="Hidden dimension.")
    parser.add_argument(
        "--freeze_embeddings",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Freeze GloVe embeddings during training.",
    )
    parser.add_argument("--dropout", type=float, default=0.3, help="Dropout probability.")
    parser.add_argument(
        "--attn_window",
        type=int,
        default=5,
        help="Attention window half-width for attention_mlp.",
    )

    # ── I/O ───────────────────────────────────────────────────────────────
    parser.add_argument(
        "--checkpoint_dir",
        default="checkpoints/",
        help="Directory to save the best model checkpoint.",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Torch device (cpu / cuda).",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=0,
        help="DataLoader worker processes.",
    )

    return parser


def main(argv: list[str] | None = None) -> int:
    """Parse arguments and run a full training run.

    Returns:
        Exit code: 0 on success, 1 on any recoverable error.
    """
    parser = build_parser()
    args = parser.parse_args(argv)

    # ── Lazy imports keep startup fast and errors surfaced clearly ────────
    from text_sentiment_classifier.config import TrainingConfig
    from text_sentiment_classifier.data.embeddings import GloVeLoader
    from text_sentiment_classifier.data.preprocessor import TextPreprocessor
    from text_sentiment_classifier.data.dataset import SentimentDataset
    from text_sentiment_classifier.factory import ModelFactory
    from text_sentiment_classifier.training.trainer import Trainer

    # ── Build and validate config ─────────────────────────────────────────
    try:
        config = TrainingConfig(
            csv_path=args.csv_path,
            glove_path=args.glove_path,
            max_len=args.max_len,
            batch_size=args.batch_size,
            epochs=args.epochs,
            lr=args.lr,
            eval_every=args.eval_every,
            model_name=args.model_name,
            hidden_dim=args.hidden_dim,
            freeze_embeddings=args.freeze_embeddings,
            dropout=args.dropout,
            attn_window=args.attn_window,
            checkpoint_dir=args.checkpoint_dir,
            device=args.device,
        )
    except ValueError as exc:
        logger.error("Configuration error: %s", exc)
        sys.exit(1)

    # Validate model name against registry.
    try:
        from text_sentiment_classifier.factory import ModelFactory as _F
        from text_sentiment_classifier.factory import _ensure_models_registered
        _ensure_models_registered()
        if args.model_name not in _F._registry:
            valid = sorted(_F._registry.keys())
            logger.error(
                "Unknown model name %r.  Valid options: %s",
                args.model_name,
                valid,
            )
            sys.exit(1)
    except Exception:
        pass  # will be caught below at ModelFactory.create

    device = torch.device(config.device)

    # ── Load GloVe embeddings ─────────────────────────────────────────────
    logger.info("Loading GloVe embeddings from %s …", config.glove_path)
    try:
        vocab, embedding_matrix = GloVeLoader().load(config.glove_path, dim=100)
    except FileNotFoundError as exc:
        logger.error(str(exc))
        logger.error(
            "Expected format: one word followed by %d floats per line, "
            "e.g.  'the 0.418 0.244 …'",
            100,
        )
        sys.exit(1)

    logger.info("Vocabulary size: %d", len(vocab))

    # ── Build datasets and data loaders ───────────────────────────────────
    preprocessor = TextPreprocessor(max_len=config.max_len, vocab=vocab)
    logger.info("Building datasets from %s …", config.csv_path)
    train_ds = SentimentDataset(config.csv_path, preprocessor, split="train")
    test_ds = SentimentDataset(config.csv_path, preprocessor, split="test")

    train_loader = DataLoader(
        train_ds,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=config.batch_size * 2,
        shuffle=False,
        num_workers=args.num_workers,
    )

    logger.info(
        "Train samples: %d   Test samples: %d", len(train_ds), len(test_ds)
    )

    # ── Create model ──────────────────────────────────────────────────────
    try:
        model = ModelFactory.create(config.model_name, embedding_matrix, config)
    except ValueError as exc:
        logger.error(str(exc))
        sys.exit(1)

    model = model.to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("Model: %s  |  trainable params: %d", config.model_name, n_params)

    # ── Train ─────────────────────────────────────────────────────────────
    trainer = Trainer(config, device=device)
    try:
        result = trainer.fit(model, train_loader, test_loader)
    except torch.cuda.OutOfMemoryError:
        logger.error(
            "CUDA out-of-memory.  Try reducing --batch_size (current: %d).",
            config.batch_size,
        )
        sys.exit(1)

    # ── Report ────────────────────────────────────────────────────────────
    logger.info(
        "Training complete.  Best accuracy: %.4f at epoch %d.",
        result.best_accuracy,
        result.best_epoch,
    )
    print(f"\nBest accuracy : {result.best_accuracy:.4f}")
    print(f"Best epoch    : {result.best_epoch}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
