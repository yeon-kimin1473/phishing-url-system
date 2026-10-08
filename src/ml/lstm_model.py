"""LSTM architecture for raw-URL character sequences.

    Input (max_len int ids)
      -> Embedding (mask_zero)
      -> LSTM
      -> Dropout
      -> Dense (ReLU)
      -> Dropout
      -> Dense(1, sigmoid)   # phishing probability

TensorFlow is imported lazily so the rest of the project (and the existing
test-suite) keeps working in environments where it is not installed.
"""

from __future__ import annotations

from src.ml.lstm_config import (
    DEFAULT_DENSE_UNITS,
    DEFAULT_DROPOUT,
    DEFAULT_EMBEDDING_DIM,
    DEFAULT_LSTM_UNITS,
)


class TensorFlowNotInstalledError(ImportError):
    """Raised when the LSTM branch is used without TensorFlow installed."""


def require_tensorflow():
    """Import and return ``tensorflow`` or raise a clear error."""
    try:
        import tensorflow as tf  # noqa: WPS433 - deliberate lazy import
    except ImportError as error:
        raise TensorFlowNotInstalledError(
            "TensorFlow is required for the LSTM branch. "
            "Install it with `pip install -r requirements.txt`."
        ) from error
    return tf


def build_lstm_model(
    vocab_size: int,
    max_len: int,
    embedding_dim: int = DEFAULT_EMBEDDING_DIM,
    lstm_units: int = DEFAULT_LSTM_UNITS,
    dense_units: int = DEFAULT_DENSE_UNITS,
    dropout: float = DEFAULT_DROPOUT,
):
    """Return a compiled Keras binary classifier."""
    tf = require_tensorflow()
    keras = tf.keras
    layers = keras.layers

    inputs = keras.Input(shape=(max_len,), dtype="int32", name="url_chars")
    x = layers.Embedding(vocab_size, embedding_dim, mask_zero=True, name="char_embedding")(inputs)
    x = layers.LSTM(lstm_units, name="lstm")(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Dense(dense_units, activation="relu")(x)
    x = layers.Dropout(dropout)(x)
    outputs = layers.Dense(1, activation="sigmoid", name="phishing_probability")(x)

    model = keras.Model(inputs, outputs, name="url_char_lstm")
    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss="binary_crossentropy",
        metrics=[keras.metrics.BinaryAccuracy(name="accuracy"), keras.metrics.AUC(name="auc")],
    )
    return model
