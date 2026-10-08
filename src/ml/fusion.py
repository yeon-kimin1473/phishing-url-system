"""Weighted probability fusion of the Random Forest and LSTM branches.

    final_probability = rf_weight * rf_probability + lstm_weight * lstm_probability
    with rf_weight + lstm_weight == 1

Weights are never assumed: :func:`select_weights` evaluates a grid on
validation data and the winner is persisted via :class:`FusionConfig`.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score, roc_auc_score,
)

from src.ml.lstm_config import FUSION_CONFIG_PATH

__all__ = [
    "FusionConfig", "FusionConfigError", "fuse", "classification_metrics",
    "select_weights", "load_fusion_config", "save_fusion_config",
]

SUPPORTED_METRICS = ("f1", "accuracy", "precision", "recall", "roc_auc")


class FusionConfigError(RuntimeError):
    """The fusion configuration is missing or invalid."""


@dataclass
class FusionConfig:
    rf_weight: float
    lstm_weight: float
    threshold: float = 0.5
    selection: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not (0.0 <= self.rf_weight <= 1.0 and 0.0 <= self.lstm_weight <= 1.0):
            raise FusionConfigError("Fusion weights must lie in [0, 1].")
        if abs(self.rf_weight + self.lstm_weight - 1.0) > 1e-6:
            raise FusionConfigError("rf_weight + lstm_weight must equal 1.")
        if not 0.0 < self.threshold < 1.0:
            raise FusionConfigError("threshold must lie in (0, 1).")


def fuse(rf_probability, lstm_probability, config: FusionConfig):
    """Return the weighted probability (works on scalars and arrays)."""
    rf = np.asarray(rf_probability, dtype=float)
    lstm = np.asarray(lstm_probability, dtype=float)
    out = config.rf_weight * rf + config.lstm_weight * lstm
    return float(out) if out.ndim == 0 else out


def classification_metrics(y_true: Sequence[int], probability: Sequence[float],
                           threshold: float = 0.5) -> Dict[str, float]:
    """Accuracy / precision / recall / F1 / ROC-AUC from real predictions."""
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(probability, dtype=float)
    pred = (p >= threshold).astype(int)
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y, p)),
    }


def select_weights(y_val, rf_val, lstm_val, metric: str = "f1",
                   step: float = 0.1, threshold: float = 0.5) -> FusionConfig:
    """Evaluate ``rf_weight`` in {0, step, ..., 1} and keep the best on ``metric``.

    Ties are broken towards the *higher* ROC-AUC, then the weight closest to
    0.5 (deterministic). All evaluated candidates are stored in
    ``config.selection['grid']`` so the choice is auditable.
    """
    if metric not in SUPPORTED_METRICS:
        raise ValueError(f"metric must be one of {SUPPORTED_METRICS}")
    n_steps = int(round(1.0 / step))
    grid: List[Dict[str, Any]] = []
    for i in range(n_steps + 1):
        w = round(i / n_steps, 6)
        fused = fuse(rf_val, lstm_val, FusionConfig(w, round(1 - w, 6), threshold))
        grid.append({"rf_weight": w, "lstm_weight": round(1 - w, 6),
                     **classification_metrics(y_val, fused, threshold)})
    best = max(grid, key=lambda r: (round(r[metric], 6), round(r["roc_auc"], 6),
                                    -abs(r["rf_weight"] - 0.5)))
    return FusionConfig(
        rf_weight=best["rf_weight"], lstm_weight=best["lstm_weight"], threshold=threshold,
        selection={"metric": metric, "validation_size": int(len(y_val)),
                   "best_validation_score": best[metric], "grid": grid},
    )


def save_fusion_config(config: FusionConfig, path: Path = FUSION_CONFIG_PATH) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(config), indent=2), encoding="utf-8")


def load_fusion_config(path: Path = FUSION_CONFIG_PATH) -> FusionConfig:
    path = Path(path)
    if not path.exists():
        raise FusionConfigError(
            f"No fusion configuration at {path}. Run `python src/ml/evaluate_hybrid.py` "
            "to select weights on validation data."
        )
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return FusionConfig(float(data["rf_weight"]), float(data["lstm_weight"]),
                            float(data.get("threshold", 0.5)), data.get("selection", {}))
    except FusionConfigError:
        raise
    except Exception as error:  # noqa: BLE001
        raise FusionConfigError(f"Invalid fusion configuration at {path}.") from error
