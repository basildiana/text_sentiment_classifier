"""Flask demo server for the text sentiment classifier.

Loads all available trained model checkpoints and exposes:
  GET  /health        — liveness check + which models are loaded
  POST /predict       — classify with one specific model
  POST /predict_all   — classify with all loaded models at once

Usage:
    # Train all four models first:
    python train.py --csv_path "IMDB Dataset.csv" --glove_path glove.6B.100d.txt --model_name gru --epochs 5
    python train.py --csv_path "IMDB Dataset.csv" --glove_path glove.6B.100d.txt --model_name rnn --epochs 5
    python train.py --csv_path "IMDB Dataset.csv" --glove_path glove.6B.100d.txt --model_name mlp --epochs 5
    python train.py --csv_path "IMDB Dataset.csv" --glove_path glove.6B.100d.txt --model_name attention_mlp --epochs 5

    # Start the demo server (loads whichever checkpoints exist):
    python app.py --glove_path glove.6B.100d.txt

    # Open demo.html in your browser (double-click the file)
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time

import torch
from flask import Flask, jsonify, request
from flask_cors import CORS

from text_sentiment_classifier.config import TrainingConfig
from text_sentiment_classifier.data.embeddings import GloVeLoader
from text_sentiment_classifier.data.preprocessor import TextPreprocessor
from text_sentiment_classifier.factory import ModelFactory

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# ---------------------------------------------------------------------------
# Globals populated at startup
# ---------------------------------------------------------------------------

# Maps model_name → loaded model instance
_models: dict = {}
_preprocessor: TextPreprocessor | None = None
_device = torch.device("cpu")

# Human-readable display names for each architecture
MODEL_DISPLAY = {
    "rnn":          {"name": "RNN",              "description": "Vanilla Recurrent Neural Network"},
    "gru":          {"name": "GRU",              "description": "Gated Recurrent Unit"},
    "mlp":          {"name": "MLP",              "description": "Token-wise Multi-Layer Perceptron"},
    "attention_mlp":{"name": "MLP + Attention",  "description": "MLP with Restricted Self-Attention"},
}


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/health", methods=["GET"])
def health():
    """Liveness check — also reports which models are loaded."""
    return jsonify({
        "status": "ok",
        "loaded_models": list(_models.keys()),
        "available_models": list(MODEL_DISPLAY.keys()),
    })


@app.route("/predict", methods=["POST"])
def predict():
    """Classify with a single model.

    Request body (JSON):
        { "text": "...", "model": "gru" }
    """
    data = request.get_json(silent=True)
    if not data or "text" not in data:
        return jsonify({"error": "Request body must be JSON with a 'text' field."}), 400

    raw_text = data["text"].strip()
    if not raw_text:
        return jsonify({"error": "'text' must not be empty."}), 400

    model_name = data.get("model", "gru")
    if model_name not in _models:
        return jsonify({
            "error": f"Model '{model_name}' is not loaded.",
            "loaded_models": list(_models.keys()),
        }), 404

    result = _run_inference(model_name, raw_text)
    return jsonify(result)


@app.route("/predict_all", methods=["POST"])
def predict_all():
    """Classify with all loaded models simultaneously.

    Request body (JSON):
        { "text": "..." }

    Response (JSON):
        {
            "results": [
                { "model_name": "gru", "display_name": "GRU", "description": "...",
                  "label": "positive", "confidence": 0.93, "probability": 0.93 },
                ...
            ]
        }
    """
    data = request.get_json(silent=True)
    if not data or "text" not in data:
        return jsonify({"error": "Request body must be JSON with a 'text' field."}), 400

    raw_text = data["text"].strip()
    if not raw_text:
        return jsonify({"error": "'text' must not be empty."}), 400

    if not _models:
        return jsonify({"error": "No models are loaded. Train at least one model first."}), 503

    results = []
    for model_name in MODEL_DISPLAY:          # preserve display order
        if model_name not in _models:
            continue
        result = _run_inference(model_name, raw_text)
        results.append(result)

    return jsonify({"results": results})


# ---------------------------------------------------------------------------
# Inference helper
# ---------------------------------------------------------------------------

def _run_inference(model_name: str, raw_text: str) -> dict:
    """Run one model and return a result dict."""
    ids = _preprocessor.process(raw_text)
    x = torch.tensor(ids, dtype=torch.long).unsqueeze(0).to(_device)

    t0 = time.perf_counter()
    with torch.no_grad():
        prob: float = _models[model_name].predict_proba(x).item()
    elapsed_ms = round((time.perf_counter() - t0) * 1000, 1)

    label = "positive" if prob > 0.5 else "negative"
    confidence = prob if label == "positive" else 1.0 - prob

    display = MODEL_DISPLAY.get(model_name, {"name": model_name, "description": ""})
    return {
        "model_name":   model_name,
        "display_name": display["name"],
        "description":  display["description"],
        "label":        label,
        "confidence":   round(confidence, 4),
        "probability":  round(prob, 4),
        "latency_ms":   elapsed_ms,
    }


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_all_models(
    glove_path: str,
    checkpoint_dir: str,
    max_len: int,
    hidden_dim: int,
) -> None:
    """Load GloVe once, then load every checkpoint that exists."""
    global _preprocessor, _device

    logger.info("Loading GloVe embeddings from %s …", glove_path)
    vocab, embedding_matrix = GloVeLoader().load(glove_path, dim=100)
    logger.info("Vocabulary size: %d", len(vocab))

    _preprocessor = TextPreprocessor(max_len=max_len, vocab=vocab)

    loaded = []
    skipped = []

    for model_name in MODEL_DISPLAY:
        ckpt_path = os.path.join(checkpoint_dir, f"{model_name}_best.pt")
        if not os.path.isfile(ckpt_path):
            skipped.append(model_name)
            continue

        try:
            config = TrainingConfig(
                glove_path=glove_path,
                max_len=max_len,
                hidden_dim=hidden_dim,
                model_name=model_name,
                freeze_embeddings=True,
            )
            model = ModelFactory.create(model_name, embedding_matrix, config)
            state_dict = torch.load(ckpt_path, map_location=_device)
            model.load_state_dict(state_dict)
            model.eval()
            model.to(_device)
            _models[model_name] = model
            loaded.append(model_name)
            logger.info("  ✓ Loaded %-15s from %s", model_name, ckpt_path)
        except Exception as exc:
            logger.warning("  ✗ Failed to load %s: %s", model_name, exc)
            skipped.append(model_name)

    if not loaded:
        logger.error(
            "No checkpoints found in '%s'. "
            "Train at least one model with train.py first.",
            checkpoint_dir,
        )
        sys.exit(1)

    if skipped:
        logger.info(
            "Skipped (no checkpoint yet): %s — train them with train.py to enable.",
            ", ".join(skipped),
        )

    logger.info("Ready with models: %s", ", ".join(loaded))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Start the multi-model sentiment classifier demo server.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--glove_path",      required=True,
                        help="Path to glove.6B.100d.txt")
    parser.add_argument("--checkpoint_dir",  default="checkpoints/",
                        help="Directory containing *_best.pt checkpoint files.")
    parser.add_argument("--max_len",         type=int, default=200)
    parser.add_argument("--hidden_dim",      type=int, default=128)
    parser.add_argument("--port",            type=int, default=5000)
    parser.add_argument("--host",            default="127.0.0.1")
    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()

    try:
        load_all_models(
            glove_path=args.glove_path,
            checkpoint_dir=args.checkpoint_dir,
            max_len=args.max_len,
            hidden_dim=args.hidden_dim,
        )
    except FileNotFoundError as exc:
        logger.error(str(exc))
        sys.exit(1)

    logger.info("Starting demo server at http://%s:%d", args.host, args.port)
    logger.info("Open demo.html in your browser to use the UI.")
    app.run(host=args.host, port=args.port, debug=False)
