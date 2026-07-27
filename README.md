# 🎬 Text Sentiment Classifier

A modular binary sentiment classifier that labels IMDB movie reviews as **positive** or **negative**.  
Supports four swappable neural architectures — RNN, GRU, MLP, and MLP with restricted self-attention — all behind a shared interface with a live browser demo.

[![Python](https://img.shields.io/badge/python-3.9+-blue.svg)]()
[![PyTorch](https://img.shields.io/badge/pytorch-2.0+-red.svg)]()
[![Accuracy](https://img.shields.io/badge/accuracy-86%25-green.svg)]()

---

## 🖥️ Live Demo (quickest way to try it)

The trained model weights are included in this repo (`checkpoints/`), so you can run the interactive demo **without retraining**.  
You only need to download one external file: the GloVe word embeddings.

### Step 1 — Download GloVe embeddings

GloVe vectors are too large for GitHub (~350 MB). Download them from Stanford:

👉 **https://nlp.stanford.edu/data/glove.6B.zip**

Unzip and place `glove.6B.100d.txt` in the project root:
```
text_sentiment_classifier/
├── glove.6B.100d.txt   ← put it here
├── checkpoints/        ← weights already here
├── demo.html
├── app.py
└── ...
```

### Step 2 — Install dependencies

```bash
python -m pip install -r requirements.txt
```

### Step 3 — Start the demo server

```bash
python app.py --glove_path glove.6B.100d.txt
```

You should see:
```
✓ Loaded rnn              from checkpoints/rnn_best.pt
✓ Loaded gru              from checkpoints/gru_best.pt
✓ Loaded mlp              from checkpoints/mlp_best.pt
✓ Loaded attention_mlp    from checkpoints/attention_mlp_best.pt
Ready with models: rnn, gru, mlp, attention_mlp
* Running on http://127.0.0.1:5000/
```

### Step 4 — Open the demo

**Double-click `demo.html`** in your file explorer (or open it in any browser as a local file).  
Type a movie review and click **Compare All Models** to see all four architectures classify it side by side.

> ⚠️ Keep the terminal running while you use the demo. The browser talks to `app.py` in the background.

---

## ✨ Features

- **Four architectures compared side-by-side** — submit one review and instantly see how RNN, GRU, MLP, and Attention MLP each respond, with confidence scores and inference speed
- **Swappable model registry** — new architectures self-register with a single decorator; the training loop needs no changes
- **Frozen GloVe embeddings** — 100-dimensional pre-trained vectors from Stanford, frozen during training for faster convergence and lower memory
- **Clean modular structure** — data pipeline, models, training, and CLI are fully decoupled
- **Property-based tests** — correctness properties verified with `hypothesis`

---

## 🧠 Architectures

| Model | Description | Test Accuracy |
|---|---|---|
| **GRU** | Gated Recurrent Unit — processes the review word-by-word, gating what to remember | ~86% |
| **RNN** | Vanilla recurrent network — simpler than GRU, faster but less accurate | ~82% |
| **MLP** | Token-wise multi-layer perceptron — scores each word independently then averages | ~85% |
| **MLP + Attention** | MLP with restricted self-attention — each word also looks at its neighbours | ~85% |

All models are trained on 40,000 IMDB reviews (80/20 train/test split) with `BCEWithLogitsLoss` and Adam optimizer.

---

## 📐 Architecture

```
text_sentiment_classifier/
├── data/
│   ├── preprocessor.py     # Text cleaning, tokenisation, padding
│   ├── embeddings.py       # GloVe file loader
│   └── dataset.py          # PyTorch Dataset (IMDB CSV)
├── models/
│   ├── base.py             # Abstract BaseClassifier
│   ├── layers.py           # BilinearLayer, RestrictedAttention
│   ├── rnn.py              # RNNClassifier
│   ├── gru.py              # GRUClassifier
│   ├── mlp.py              # MLPClassifier
│   └── attention_mlp.py    # AttentionMLPClassifier
├── training/
│   ├── trainer.py          # Training loop (BCE + Adam)
│   └── checkpointer.py     # Saves best checkpoint per model
├── utils/
│   └── metrics.py          # Accuracy, confusion matrix
├── config.py               # TrainingConfig, EvalResult dataclasses
├── factory.py              # ModelFactory (decorator-based registry)
├── train.py                # CLI for training
└── app.py                  # Flask demo server
```

---

## 🔁 Retrain from Scratch (optional)

If you want to retrain the models yourself, you also need the dataset:

👉 **https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews**

Download `IMDB Dataset.csv` and place it in the project root.  
Then run one command per architecture (~25 min each on CPU):

```bash
python train.py --csv_path "IMDB Dataset.csv" --glove_path glove.6B.100d.txt --model_name gru --epochs 5
python train.py --csv_path "IMDB Dataset.csv" --glove_path glove.6B.100d.txt --model_name rnn --epochs 5
python train.py --csv_path "IMDB Dataset.csv" --glove_path glove.6B.100d.txt --model_name mlp --epochs 5
python train.py --csv_path "IMDB Dataset.csv" --glove_path glove.6B.100d.txt --model_name attention_mlp --epochs 5
```

Checkpoints are saved automatically to `checkpoints/` whenever a new best accuracy is reached.

---

## 🔬 Running Tests

```bash
python -m pytest tests/ -v
```

---

## 📦 Dependencies

```
torch>=2.0.0
numpy>=1.24.0
pandas>=2.0.0
flask>=3.0.0
flask-cors>=4.0.0
hypothesis>=6.0.0
pytest>=7.0.0
```
