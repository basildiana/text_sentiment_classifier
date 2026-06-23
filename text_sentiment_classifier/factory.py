"""ModelFactory: registry-based factory for classifier architectures.

New architectures self-register via the ``@ModelFactory.register`` decorator::

    @ModelFactory.register("my_model")
    class MyModel(BaseClassifier):
        ...

The four built-in architectures (``rnn``, ``gru``, ``mlp``, ``attention_mlp``)
are registered by importing their modules at the bottom of this file.
"""

from __future__ import annotations

from typing import Callable, Dict, Type

import torch

from text_sentiment_classifier.config import TrainingConfig
from text_sentiment_classifier.models.base import BaseClassifier


class ModelFactory:
    """Maps string model names to concrete :class:`BaseClassifier` subclasses.

    Usage::

        model = ModelFactory.create("gru", embedding_matrix, config)
    """

    _registry: Dict[str, Type[BaseClassifier]] = {}

    @classmethod
    def register(cls, name: str) -> Callable[[Type[BaseClassifier]], Type[BaseClassifier]]:
        """Decorator that registers a classifier class under ``name``.

        Args:
            name: The string key used to look up this architecture.

        Returns:
            A decorator that adds the class to the registry and returns it unchanged.
        """

        def decorator(klass: Type[BaseClassifier]) -> Type[BaseClassifier]:
            cls._registry[name] = klass
            return klass

        return decorator

    @classmethod
    def create(
        cls,
        name: str,
        embedding_matrix: torch.Tensor,
        config: TrainingConfig,
    ) -> BaseClassifier:
        """Instantiate the classifier registered under ``name``.

        Args:
            name:             Registered architecture name.
            embedding_matrix: Pre-trained embedding tensor ``[V, E]``.
            config:           Training / model configuration.

        Returns:
            A new instance of the corresponding :class:`BaseClassifier` subclass.

        Raises:
            ValueError: If ``name`` is not registered, with a descriptive message
                        listing all valid names.
        """
        # Ensure all built-in models are loaded before looking up the registry.
        _ensure_models_registered()

        if name not in cls._registry:
            available = sorted(cls._registry.keys())
            raise ValueError(
                f"Unknown model {name!r}.  "
                f"Available: {available}"
            )
        klass = cls._registry[name]
        return klass(embedding_matrix, config)


_models_registered = False


def _ensure_models_registered() -> None:
    """Import built-in model modules exactly once to trigger @register decorators."""
    global _models_registered
    if _models_registered:
        return
    _models_registered = True
    # These imports are intentionally deferred to break the circular import that
    # would occur if model modules were imported at module-level of factory.py
    # (model modules import ModelFactory, factory.py would import models → cycle).
    import text_sentiment_classifier.models.rnn  # noqa: F401
    import text_sentiment_classifier.models.gru  # noqa: F401
    import text_sentiment_classifier.models.mlp  # noqa: F401
    import text_sentiment_classifier.models.attention_mlp  # noqa: F401
