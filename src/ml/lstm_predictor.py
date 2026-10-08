"""Inference for the LSTM branch: raw URL in, phishing probability out.

Loads the model and the tokenizer written by ``train_lstm.py``. If either is
missing, a clear :class:`LSTMNotAvailableError` is raised; nothing is guessed.
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Any, Dict

import numpy as np

from src.ml.char_tokenizer import CharTokenizer
from src.ml.lstm_config import LSTM_MODEL_PATH, LSTM_TOKENIZER_PATH
from src.ml.lstm_model import TensorFlowNotInstalledError, require_tensorflow

LOGGER = logging.getLogger(__name__)

__all__ = ["LSTMNotAvailableError", "LSTMPredictor", "predict_lstm"]


class LSTMNotAvailableError(RuntimeError):
    """The LSTM model/tokenizer is missing, unreadable or TensorFlow is absent."""


class LSTMPredictor:
    """Loads the LSTM once and scores URLs with the training-time tokenizer."""

    def __init__(
        self,
        model_path: Path = LSTM_MODEL_PATH,
        tokenizer_path: Path = LSTM_TOKENIZER_PATH,
        threshold: float = 0.5,
    ) -> None:
        self.model_path = Path(model_path)
        self.tokenizer_path = Path(tokenizer_path)
        self.threshold = threshold
        self._model = None
        self._tokenizer: CharTokenizer | None = None
        self._lock = threading.Lock()

    @property
    def is_ready(self) -> bool:
        """``True`` when both artefacts exist (TensorFlow is checked on load)."""
        return self.model_path.exists() and self.tokenizer_path.exists()

    def load(self) -> None:
        if self._model is not None:
            return
        with self._lock:
            if self._model is not None:
                return
            if not self.is_ready:
                raise LSTMNotAvailableError(
                    f"LSTM artefacts not found ({self.model_path}, {self.tokenizer_path}). "
                    "The LSTM must be trained first: `python src/ml/train_lstm.py`."
                )
            try:
                tf = require_tensorflow()
                tokenizer = CharTokenizer.load(self.tokenizer_path)
                model = tf.keras.models.load_model(self.model_path, compile=False)
            except TensorFlowNotInstalledError as error:
                raise LSTMNotAvailableError(str(error)) from error
            except Exception as error:  # noqa: BLE001
                LOGGER.exception("Failed to load LSTM artefacts")
                raise LSTMNotAvailableError(
                    "The LSTM model could not be loaded. Please retrain it."
                ) from error
            self._tokenizer, self._model = tokenizer, model

    def predict_proba(self, url: str) -> float:
        """Return the LSTM's phishing probability for one URL."""
        self.load()
        assert self._tokenizer is not None
        encoded = self._tokenizer.encode_batch([url])
        return float(np.asarray(self._model(encoded, training=False)).reshape(-1)[0])

    def predict(self, url: str) -> Dict[str, Any]:
        probability = self.predict_proba(url)
        return {
            "prediction": int(probability >= self.threshold),
            "probability": probability,
        }


_default: LSTMPredictor | None = None
_default_lock = threading.Lock()


def predict_lstm(url: str) -> Dict[str, Any]:
    """Score ``url`` with the default LSTM artefacts.

    Returns ``{"prediction": 0 | 1, "probability": float}`` where
    ``probability`` is the LSTM's phishing probability.
    """
    global _default
    with _default_lock:
        if _default is None:
            _default = LSTMPredictor()
    return _default.predict(url)
