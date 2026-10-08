"""Hybrid RF + LSTM prediction.

Uses the EXISTING :class:`src.ml.predictor.PhishingPredictor` for the feature
branch (nothing is copied from it), the LSTM predictor for the character
branch, and :mod:`src.ml.fusion` to combine them.

Explanations: the ``rf_result`` entry carries the existing feature-based
counterfactual explanation. It explains the Random Forest branch only; the
LSTM contributes a probability and is not explained by it.
"""

from __future__ import annotations

from typing import Any, Dict

from src.ml.fusion import FusionConfig, fuse, load_fusion_config
from src.ml.lstm_predictor import LSTMPredictor
from src.ml.predictor import PhishingPredictor

__all__ = ["HybridPredictor", "predict_hybrid"]


class HybridPredictor:
    def __init__(self, rf_predictor: PhishingPredictor, lstm_predictor: LSTMPredictor | None = None,
                 fusion_config: FusionConfig | None = None) -> None:
        self.rf_predictor = rf_predictor
        self.lstm_predictor = lstm_predictor or LSTMPredictor()
        self._fusion_config = fusion_config

    @property
    def fusion_config(self) -> FusionConfig:
        if self._fusion_config is None:
            self._fusion_config = load_fusion_config()
        return self._fusion_config

    def predict(self, url: str) -> Dict[str, Any]:
        """Score ``url`` with RF and LSTM and fuse the probabilities.

        Raises ``ModelNotAvailableError`` (RF), ``LSTMNotAvailableError`` or
        ``FusionConfigError`` if a component is missing; never fabricates.
        """
        config = self.fusion_config  # fail early if weights are missing
        rf_result = self.rf_predictor.predict(url)
        lstm_probability = self.lstm_predictor.predict_proba(url)
        rf_probability = float(rf_result.phishing_probability)

        probability = float(fuse(rf_probability, lstm_probability, config))
        prediction = int(probability >= config.threshold)
        rf_pred = self.rf_predictor
        verdict = ("Phishing" if probability >= rf_pred.phishing_threshold
                   else "Suspicious" if probability >= rf_pred.suspicious_threshold else "Safe")
        return {
            "url": url,
            "prediction": prediction,
            "label": "Phishing" if prediction else "Legitimate",
            "verdict": verdict,  # same thresholds the existing app uses
            "probability": round(probability, 6),
            "rf_probability": round(rf_probability, 6),
            "lstm_probability": round(float(lstm_probability), 6),
            "weights": {"rf": config.rf_weight, "lstm": config.lstm_weight},
            "rf_explanation": {  # Random Forest feature branch only
                "contributions": rf_result.contributions,
                "indicators": rf_result.indicators,
            },
        }


def predict_hybrid(url: str, rf_predictor: PhishingPredictor | None = None) -> Dict[str, Any]:
    """Convenience wrapper using the configured default artefacts."""
    if rf_predictor is None:
        from config import BaseConfig
        rf_predictor = PhishingPredictor(
            BaseConfig.MODEL_PATH, BaseConfig.SUSPICIOUS_THRESHOLD, BaseConfig.PHISHING_THRESHOLD)
    return HybridPredictor(rf_predictor).predict(url)
