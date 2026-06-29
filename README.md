# Text Sentiment Classifier
A modular binary sentiment classifier designed to evaluate reviews from the IMDB dataset as either **positive** or **negative**. 

[![Status](https://img.shields.io/badge/status-in__progress-orange.svg)]()
[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)]()
[![PyTorch](https://img.shields.io/badge/pytorch-2.0+-red.svg)]()


> ⚠️ **Project Status: Work in Progress** > This repo contains design documentation, architectural specifications, and core interfaces. 
Active development of the underlying PyTorch training pipelines and model architectures is currently underway.
---

## 🚀 Key Features 

* **Swappable Architectures:** Implements a unified interface supporting multiple underlying models behind a strict registry pattern (RNN, GRU, Token-wise MLP, and an MLP utilizing restricted self-attention).
* **Decoupled Design:** Features a dedicated `ModelFactory` to cleanly isolate training orchestration from specific neural network configurations.
* **Pre-trained Embeddings:** Leverages Stanford's GloVe word embeddings (`glove.6B.100d`) with a custom execution pipeline handling unknown vocabulary tokens gracefully.
* **Algorithmic Discipline:** Built with strict software engineering constraints, utilizing comprehensive preconditions, postconditions, and property-based testing principles.

---

## 📐 System Architecture

The project decouples components to allow seamless swapping of dataset loaders, token preprocessors, and evaluation loops.

```mermaid
graph TD
    A[CLI / train.py entrypoint] --> B[Trainer]
    B --> C[SentimentDataset]
    B --> D[ModelFactory]
    B --> E[Checkpointer]
    C --> F[TextPreprocessor]
    C --> G[GloVeLoader]
    D --> H[RNNClassifier]
    D --> I[GRUClassifier]
    D --> J[MLPClassifier]
    D --> K[AttentionMLPClassifier]
    H & I & J & K --> L[BaseClassifier]
    L --> M[BilinearLayer]
