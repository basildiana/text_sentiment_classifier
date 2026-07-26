"""Flask demo server for the text sentiment classifier.

Exposes a single POST /predict endpoint that accepts raw review text and
returns the predicted sentiment label and confidence score.

Usage:
    # 1. Train the model first (produces checkpoints/gru_best.pt)
    python train.py --csv_path "IMDB Dataset.csv" --glove_path glove.6B.100d.txt

    # 2. Start the demo server
    python app.py --checkpoint checkpoints/gru_best.pt \
                  --glove_path glove.6B.100d.txt \
                  --model_name gru

    # 3. Open demo.html in your browser
"""

from __future__ import annotations

import argparse
import json
import logging
import sys

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
CORS(app)  # Allow the HTML file to call this API from any origin

# ---------------------------------------------------------------------------
# Globals populated at startup
# ---------------------------------------------------------------------------
_model = None
_preprocessor: TextPreprocessor | None = None
_device = torch.device("cpu")


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/health", methods=["GET"])
def health():
    """Simple liveness check."""
    return jsonify({"status": "ok"})


@app.route("/predict", methods=["POST"])
def predict():
    """Classify the sentiment of a review.

    Request body (JSON):
        { "text": "This movie was fantastic!" }

    Response (JSON):
        {
            "label":       "positive",
            "confidence":  0.93,
            "probability": 0.93
        }
    """
    if _model is None or _preprocessor is None:
        return jsonify({"error": "Model not loaded"}), 503

    data = request.get_json(silent=True)
    if not data or "text" not in data:
        return jsonify({"error": "Request body must be JSON with a 'text' field."}), 400

    raw_text: str = data["text"].strip()
    if not raw_text:
        return jsonify({"error": "'text' field must not be empty."}), 400

    # Preprocess → tensor → forward → probability
    ids = _preprocessor.process(raw_text)
    x = torch.tensor(ids, dtype=torch.long).unsqueeze(0).to(_device)  # [1, L]

    with torch.no_grad():
        prob: float = _model.predict_proba(x).item()

    label = "positive" if prob > 0.5 else "negative"
    confidence = prob if label == "positive" else 1.0 - prob

    return jsonify({
        "label":       label,
        "confidence":  round(confidence, 4),
        "probability": round(prob, 4),
    })


# ---------------------------------------------------------------------------
# Startup
# ---------------------------------------------------------------------------

def load_model(
    checkpoint_path: str,
    glove_path: str,
    model_name: str,
    max_len: int,
    hidden_dim: int,
) -> None:
    """Load GloVe embeddings and restore model weights from a checkpoint."""
    global _model, _preprocessor, _device

    logger.info("Loading GloVe embeddings from %s …", glove_path)
    vocab, embedding_matrix = GloVeLoader().load(glove_path, dim=100)
    logger.info("Vocabulary size: %d", len(vocab))

    _preprocessor = TextPreprocessor(max_len=max_len, vocab=vocab)

    config = TrainingConfig(
        glove_path=glove_path,
        max_len=max_len,
        hidden_dim=hidden_dim,
        model_name=model_name,
        freeze_embeddings=True,
    )

    _model = ModelFactory.create(model_name, embedding_matrix, config)
    state_dict = torch.load(checkpoint_path, map_location=_device)
    _model.load_state_dict(state_dict)
    _model.eval()
    _model.to(_device)

    logger.info("Model '%s' loaded from %s", model_name, checkpoint_path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Start the sentiment classifier demo server.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Path to a saved model checkpoint (.pt file).",
    )
    parser.add_argument(
        "--glove_path",
        required=True,
        help="Path to GloVe embeddings file (glove.6B.100d.txt).",
    )
    parser.add_argument(
        "--model_name",
        default="gru",
        help="Architecture that was used during training: rnn | gru | mlp | attention_mlp.",
    )
    parser.add_argument("--max_len", type=int, default=200)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--host", default="127.0.0.1")
    return parser


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()

    try:
        load_model(
            checkpoint_path=args.checkpoint,
            glove_path=args.glove_path,
            model_name=args.model_name,
            max_len=args.max_len,
            hidden_dim=args.hidden_dim,
        )
    except FileNotFoundError as exc:
        logger.error(str(exc))
        sys.exit(1)
    except Exception as exc:
        logger.error("Failed to load model: %s", exc)
        sys.exit(1)

    logger.info("Starting demo server at http://%s:%d", args.host, args.port)
    logger.info("Open demo.html in your browser to use the UI.")
    app.run(host=args.host, port=args.port, debug=False)
