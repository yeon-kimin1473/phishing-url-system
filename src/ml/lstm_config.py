"""Paths and defaults for the LSTM branch and the hybrid fusion.

Kept separate from the project-wide ``config.py`` so the existing
configuration is untouched. Every path can be overridden by environment
variable, mirroring how ``config.py`` handles ``MODEL_PATH``.
"""

from __future__ import annotations

import os
from pathlib import Path

from config import BaseConfig

MODEL_DIR: Path = BaseConfig.MODEL_DIR

LSTM_MODEL_PATH: Path = Path(os.getenv("LSTM_MODEL_PATH", MODEL_DIR / "lstm_model.keras"))
LSTM_TOKENIZER_PATH: Path = Path(os.getenv("LSTM_TOKENIZER_PATH", MODEL_DIR / "lstm_tokenizer.json"))
LSTM_METRICS_PATH: Path = Path(os.getenv("LSTM_METRICS_PATH", MODEL_DIR / "lstm_metrics.json"))
FUSION_CONFIG_PATH: Path = Path(os.getenv("FUSION_CONFIG_PATH", MODEL_DIR / "fusion_config.json"))
COMPARISON_PATH: Path = Path(os.getenv("COMPARISON_PATH", MODEL_DIR / "model_comparison.json"))

#: Character-sequence length. ~98% of the bundled corpus is <= 200 characters.
DEFAULT_MAX_LEN = 200
DEFAULT_EMBEDDING_DIM = 32
DEFAULT_LSTM_UNITS = 64
DEFAULT_DENSE_UNITS = 32
DEFAULT_DROPOUT = 0.3

#: Split parameters shared by RF training (src/ml/train.py defaults) and the
#: LSTM, so both models are evaluated on the *same* held-out URLs.
DEFAULT_SEED = 42
DEFAULT_TEST_SIZE = 0.2
