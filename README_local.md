# Text Sentiment Classifier

A portfolio-quality binary sentiment classifier (positive / negative) trained on IMDB movie reviews.
Built with PyTorch and pre-trained GloVe embeddings; four swappable architectures — RNN, GRU,
token-wise MLP, and Attention MLP — all behind a shared interface.

---

## Project Structure

```
text_sentiment_classifier/
├── data/
│   ├── preprocessor.py     # TextPreprocessor — clean → tokenize → encode → pad
│   ├── embeddings.py       # GloVeLoader — loads pre-trained GloVe vectors
│   └── dataset.py          # SentimentDataset — PyTorch Dataset over IMDB CSV
├── models/
│   ├── base.py             # BaseClassifier — abstract base class
│   ├── layers.py           # BilinearLayer, RestrictedAttention
│   ├── rnn.py              # RNNClassifier
│   ├── gru.py              # GRUClassifier
│   ├── mlp.py              # MLPClassifier
│   └── attention_mlp.py    # AttentionMLPClassifier
├── training/
│   ├── trainer.py          # Trainer — training loop + evaluation
│   └── checkpointer.py     # Checkpointer — save best weights
├── utils/
│   └── metrics.py          # compute_accuracy, compute_confusion
├── config.py               # TrainingConfig, EvalResult, TrainingResult
└── factory.py              # ModelFactory — registry-based model creation
train.py                    # CLI entrypoint
tests/                      # Unit + property-based tests (hypothesis + pytest)
requirements.txt
```

---

## Installation

```bash
pip install -r requirements.txt
```

### Download GloVe Vectors

Download pre-trained GloVe embeddings from the [Stanford NLP GloVe page](https://nlp.stanford.edu/projects/glove/)
and extract `glove.6B.100d.txt` (or another dimension file) to a location of your choice.

```
wget https://downloads.cs.stanford.edu/nlp/data/glove.6B.zip
unzip glove.6B.zip
```

### Dataset

The IMDB CSV file must have three columns: `review`, `sentiment` (`"positive"` / `"negative"`),
and `split` (`"train"` / `"test"`).

---

## Training

### GRU (default)

```bash
python train.py \
    --csv_path data/imdb.csv \
    --glove_path glove.6B.100d.txt \
    --model_name gru \
    --epochs 10 \
    --batch_size 32 \
    --hidden_dim 128
```

### RNN

```bash
python train.py \
    --csv_path data/imdb.csv \
    --glove_path glove.6B.100d.txt \
    --model_name rnn \
    --epochs 10 \
    --hidden_dim 128
```

### Token-wise MLP

```bash
python train.py \
    --csv_path data/imdb.csv \
    --glove_path glove.6B.100d.txt \
    --model_name mlp \
    --epochs 10 \
    --hidden_dim 256
```

### Attention MLP

```bash
python train.py \
    --csv_path data/imdb.csv \
    --glove_path glove.6B.100d.txt \
    --model_name attention_mlp \
    --epochs 10 \
    --hidden_dim 128 \
    --attn_window 5
```

### All options

```
python train.py --help
```

---

## Inference

```python
import torch
from text_sentiment_classifier.data.embeddings import GloVeLoader
from text_sentiment_classifier.data.preprocessor import TextPreprocessor
from text_sentiment_classifier.config import TrainingConfig
from text_sentiment_classifier.factory import ModelFactory
from text_sentiment_classifier.training.checkpointer import Checkpointer

# 1. Load embeddings
vocab, embedding_matrix = GloVeLoader().load("glove.6B.100d.txt", dim=100)

# 2. Rebuild the same config used during training
config = TrainingConfig(
    csv_path="data/imdb.csv",
    glove_path="glove.6B.100d.txt",
    model_name="gru",
    hidden_dim=128,
)

# 3. Create and load the best checkpoint
model = ModelFactory.create(config.model_name, embedding_matrix, config)
ckpt = Checkpointer(save_dir="checkpoints/", model_name=config.model_name)
model = ckpt.load_best(model)
model.eval()

# 4. Classify a raw review
preprocessor = TextPreprocessor(max_len=config.max_len, vocab=vocab)
raw = "This movie was absolutely fantastic — great performances all around."
ids = preprocessor.process(raw)
x   = torch.tensor(ids).unsqueeze(0)   # [1, max_len]
with torch.no_grad():
    prob = torch.sigmoid(model(x)).item()
label = "positive" if prob > 0.5 else "negative"
print(f"{label} ({prob:.2%})")
```

---

## Running Tests

```bash
pytest tests/ -v
```

Property-based tests use the [Hypothesis](https://hypothesis.readthedocs.io/) library and verify
seven correctness properties across a large number of randomly generated inputs.

---

## Model Architectures

| Name | Description |
|---|---|
| `rnn` | Single-layer vanilla RNN, final hidden state → linear head |
| `gru` | Single-layer GRU (gated recurrent unit), final hidden state → linear head |
| `mlp` | Token-wise linear projection, mean-pooled → linear head |
| `attention_mlp` | Token-wise MLP + window-restricted self-attention → linear head |

All architectures share the same `BaseClassifier` interface:
- Input:  `LongTensor [batch, seq_len]`
- Output: `FloatTensor [batch, 1]` (raw logits; apply `sigmoid` for probability)

---

## Design

See [`design.md`](design.md) for the full architecture specification, sequence diagrams,
algorithmic pseudocode, and formal correctness properties.
