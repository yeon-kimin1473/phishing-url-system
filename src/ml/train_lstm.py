"""Train the character-level LSTM on RAW URL + LABEL.

    python src/ml/train_lstm.py                  # full dataset
    python src/ml/train_lstm.py --sample 20000   # quick run
    python src/ml/train_lstm.py --epochs 5

Dataset loading reuses ``load_dataset`` from ``src/ml/train.py`` (imported, not
modified), so the URL/label columns are detected by exactly the same rules as
the Random Forest pipeline. No handcrafted features are used.

Split design (important for an honest hybrid evaluation)
-------------------------------------------------------
The RF in ``train.py`` is fitted on 80% of the rows and ``--seed``/``--test-size``
define the other 20%. This script reproduces that exact partition
(``make_splits``) and trains the LSTM only on the same 80% (a further 10% of it
is used for early stopping). The 20% hold-out is therefore unseen by BOTH
models, and ``evaluate_hybrid.py`` splits it again into a fusion-validation half
(weight selection) and a final-test half (reporting).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from config import BaseConfig
from src.ml.char_tokenizer import CharTokenizer
from src.ml.fusion import classification_metrics
from src.ml.lstm_config import (
    DEFAULT_MAX_LEN, DEFAULT_SEED, DEFAULT_TEST_SIZE,
    LSTM_METRICS_PATH, LSTM_MODEL_PATH, LSTM_TOKENIZER_PATH,
)
from src.ml.lstm_model import TensorFlowNotInstalledError, build_lstm_model, require_tensorflow
from src.ml.train import load_dataset

LOGGER = logging.getLogger("train_lstm")


def set_seeds(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    require_tensorflow().keras.utils.set_random_seed(seed)


def load_frame(dataset_path: Path, sample: int | None, seed: int) -> pd.DataFrame:
    """Load/clean exactly like ``train.py`` (incl. the optional stratified sample)."""
    frame = load_dataset(dataset_path)
    if sample is not None and sample < len(frame):
        frame, _ = train_test_split(frame, train_size=sample, stratify=frame["label"],
                                    random_state=seed)
        frame = frame.reset_index(drop=True)
    return frame


def make_splits(labels: Sequence[int], seed: int, test_size: float) -> Dict[str, np.ndarray]:
    """Return index arrays: ``train`` / ``lstm_val`` / ``fusion_val`` / ``test``.

    ``train`` + ``lstm_val`` == the RF training rows; ``fusion_val`` + ``test``
    == the RF test rows (same indices as ``train.py``'s split).
    """
    y = np.asarray(labels, dtype=int)
    index = np.arange(len(y))
    rf_train, holdout = train_test_split(index, test_size=test_size, stratify=y, random_state=seed)
    train, lstm_val = train_test_split(rf_train, test_size=0.1, stratify=y[rf_train], random_state=seed)
    fusion_val, test = train_test_split(holdout, test_size=0.5, stratify=y[holdout], random_state=seed)
    return {"train": train, "lstm_val": lstm_val, "fusion_val": fusion_val, "test": test}


def train(dataset_path: Path, model_path: Path, tokenizer_path: Path, metrics_path: Path,
          max_len: int = DEFAULT_MAX_LEN, epochs: int = 10, batch_size: int = 256,
          seed: int = DEFAULT_SEED, test_size: float = DEFAULT_TEST_SIZE,
          sample: int | None = None) -> Dict[str, Any]:
    tf = require_tensorflow()
    set_seeds(seed)

    frame = load_frame(dataset_path, sample, seed)
    urls = frame["url"].tolist()
    labels = frame["label"].to_numpy(dtype=int)
    splits = make_splits(labels, seed, test_size)
    print(f"Rows: {len(frame):,} | LSTM train {len(splits['train']):,} | "
          f"early-stop val {len(splits['lstm_val']):,} | hold-out {len(splits['fusion_val']) + len(splits['test']):,}")

    # Vocabulary from the TRAINING rows only (no leakage from validation/test).
    tokenizer = CharTokenizer.fit([urls[i] for i in splits["train"]], max_len=max_len)
    def encode(idx): return tokenizer.encode_batch([urls[i] for i in idx])
    x_train, x_val = encode(splits["train"]), encode(splits["lstm_val"])
    y_train, y_val = labels[splits["train"]], labels[splits["lstm_val"]]

    model = build_lstm_model(tokenizer.vocab_size, max_len)
    model.summary(print_fn=print)
    history = model.fit(
        x_train, y_train, validation_data=(x_val, y_val), epochs=epochs, batch_size=batch_size,
        callbacks=[tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=2,
                                                    restore_best_weights=True)],
        verbose=2,
    )

    holdout = np.concatenate([splits["fusion_val"], splits["test"]])
    probs = model.predict(encode(holdout), batch_size=1024, verbose=0).reshape(-1)
    metrics = classification_metrics(labels[holdout], probs)
    print("\nLSTM metrics on the 20% hold-out (unseen in training and early stopping):")
    for name, value in metrics.items():
        print(f"  {name:<10}: {value:.4f}")

    model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(model_path)
    tokenizer.save(tokenizer_path)
    document = {
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "dataset": {"path": str(dataset_path), "rows": int(len(frame)),
                    "sample": sample, "seed": seed, "test_size": test_size},
        "max_len": max_len, "vocab_size": tokenizer.vocab_size,
        "epochs_run": len(history.history["loss"]),
        "holdout_metrics": metrics,
        "tensorflow_version": tf.__version__,
    }
    metrics_path.write_text(json.dumps(document, indent=2), encoding="utf-8")
    print(f"\nSaved: {model_path}\n       {tokenizer_path}\n       {metrics_path}")
    return document


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train the URL character LSTM.",
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--dataset", type=Path, default=BaseConfig.DATASET_PATH)
    p.add_argument("--output", type=Path, default=LSTM_MODEL_PATH)
    p.add_argument("--tokenizer-output", type=Path, default=LSTM_TOKENIZER_PATH)
    p.add_argument("--metrics-output", type=Path, default=LSTM_METRICS_PATH)
    p.add_argument("--max-len", type=int, default=DEFAULT_MAX_LEN)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Must match src/ml/train.py")
    p.add_argument("--test-size", type=float, default=DEFAULT_TEST_SIZE, help="Must match src/ml/train.py")
    p.add_argument("--sample", type=int, default=None, help="Must match src/ml/train.py --sample if used")
    return p.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s")
    args = parse_args(argv)
    try:
        train(args.dataset, args.output, args.tokenizer_output, args.metrics_output,
              args.max_len, args.epochs, args.batch_size, args.seed, args.test_size, args.sample)
    except TensorFlowNotInstalledError as error:
        print(f"\n[ERROR] {error}\n", file=sys.stderr)
        return 4
    except (FileNotFoundError, ValueError) as error:
        print(f"\n[ERROR] {error}\n", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
