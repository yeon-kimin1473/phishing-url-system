"""Select fusion weights on validation data and compare RF vs LSTM vs Hybrid.

    python src/ml/evaluate_hybrid.py [--metric f1] [--sample N]

Prerequisites: ``python src/ml/train.py`` and ``python src/ml/train_lstm.py``
(run both with the same --seed/--test-size/--sample).

Protocol
--------
* Both models were trained without the 20% hold-out (see ``train_lstm.py``).
* The hold-out is split into ``fusion_val`` (weights chosen here) and ``test``
  (all three models reported here; never used for any selection).
* Every number written to ``models/model_comparison.json`` is computed in this
  run from real model outputs.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import joblib
import numpy as np

from config import BaseConfig
from src.features.url_features import FEATURE_NAMES
from src.ml.char_tokenizer import CharTokenizer
from src.ml.fusion import (
    SUPPORTED_METRICS, classification_metrics, fuse, save_fusion_config, select_weights,
)
from src.ml.lstm_config import (
    COMPARISON_PATH, DEFAULT_SEED, DEFAULT_TEST_SIZE, FUSION_CONFIG_PATH,
    LSTM_MODEL_PATH, LSTM_TOKENIZER_PATH,
)
from src.ml.lstm_model import require_tensorflow
from src.ml.train import build_feature_matrix
from src.ml.train_lstm import load_frame, make_splits


def rf_probabilities(bundle: Dict[str, Any], urls: Sequence[str]) -> np.ndarray:
    """Batch phishing probabilities from the EXISTING trained bundle."""
    matrix = build_feature_matrix(urls)[list(bundle["feature_names"])].to_numpy(dtype=float)
    if bundle.get("requires_scaling") and bundle.get("scaler") is not None:
        matrix = bundle["scaler"].transform(matrix)
    return bundle["model"].predict_proba(matrix)[:, 1]


def evaluate(dataset: Path, metric: str, seed: int, test_size: float, sample: int | None,
             step: float) -> Dict[str, Any]:
    if not BaseConfig.MODEL_PATH.exists():
        raise FileNotFoundError(f"No RF model at {BaseConfig.MODEL_PATH}. Run `python src/ml/train.py`.")
    if not (LSTM_MODEL_PATH.exists() and LSTM_TOKENIZER_PATH.exists()):
        raise FileNotFoundError("LSTM artefacts missing. Run `python src/ml/train_lstm.py` first.")

    bundle = joblib.load(BaseConfig.MODEL_PATH)
    frame = load_frame(dataset, sample, seed)
    rf_meta = bundle.get("dataset", {})
    if (rf_meta.get("rows") != len(frame) or rf_meta.get("random_state") != seed
            or rf_meta.get("test_size") != test_size):
        raise ValueError(
            "The RF bundle was trained on a different dataset/split than the one being "
            f"evaluated (RF: {rf_meta}; here: rows={len(frame)}, seed={seed}, test_size={test_size}). "
            "Use matching --seed/--test-size/--sample so the hold-out is unseen by both models.")

    urls = frame["url"].tolist()
    labels = frame["label"].to_numpy(dtype=int)
    splits = make_splits(labels, seed, test_size)

    tf = require_tensorflow()
    tokenizer = CharTokenizer.load(LSTM_TOKENIZER_PATH)
    lstm = tf.keras.models.load_model(LSTM_MODEL_PATH, compile=False)

    def probs(idx):
        batch = [urls[i] for i in idx]
        return (rf_probabilities(bundle, batch),
                lstm.predict(tokenizer.encode_batch(batch), batch_size=1024, verbose=0).reshape(-1))

    rf_val, lstm_val = probs(splits["fusion_val"])
    rf_test, lstm_test = probs(splits["test"])
    y_val, y_test = labels[splits["fusion_val"]], labels[splits["test"]]

    config = select_weights(y_val, rf_val, lstm_val, metric=metric, step=step)
    save_fusion_config(config, FUSION_CONFIG_PATH)
    hybrid_test = fuse(rf_test, lstm_test, config)

    report = {
        "evaluated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "rf_model_name": bundle.get("model_name"),
        "protocol": "weights chosen on fusion_val; metrics below are on the untouched test split",
        "selection_metric": metric,
        "selected_weights": {"rf": config.rf_weight, "lstm": config.lstm_weight},
        "fusion_val_size": int(len(y_val)), "test_size": int(len(y_test)),
        "test_metrics": {
            "random_forest": classification_metrics(y_test, rf_test),
            "lstm": classification_metrics(y_test, lstm_test),
            "hybrid": classification_metrics(y_test, hybrid_test),
        },
        "validation_grid": config.selection["grid"],
    }
    COMPARISON_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def print_report(report: Dict[str, Any]) -> None:
    print(f"\nRF model: {report['rf_model_name']} | selection metric: {report['selection_metric']} "
          f"(on {report['fusion_val_size']:,} validation rows)")
    print(f"Selected weights: RF={report['selected_weights']['rf']}, LSTM={report['selected_weights']['lstm']}")
    print(f"\nTest split ({report['test_size']:,} rows):")
    print(f"{'model':<14}" + "".join(f"{m:>11}" for m in SUPPORTED_METRICS))
    for name, m in report["test_metrics"].items():
        print(f"{name:<14}" + "".join(f"{m[k]:>11.4f}" for k in SUPPORTED_METRICS))


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", type=Path, default=BaseConfig.DATASET_PATH)
    p.add_argument("--metric", choices=SUPPORTED_METRICS, default="f1")
    p.add_argument("--step", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--test-size", type=float, default=DEFAULT_TEST_SIZE)
    p.add_argument("--sample", type=int, default=None)
    args = p.parse_args(argv)
    try:
        print_report(evaluate(args.dataset, args.metric, args.seed, args.test_size, args.sample, args.step))
    except (FileNotFoundError, ValueError, ImportError) as error:
        print(f"\n[ERROR] {error}\n", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
