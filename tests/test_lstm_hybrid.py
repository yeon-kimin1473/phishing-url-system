"""Tests for the LSTM preprocessing, fusion, hybrid predictor and error paths.

Tests that need a *trained* LSTM are skipped until ``train_lstm.py`` has been
run and TensorFlow is installed. The hybrid-plumbing tests inject stand-in
branch predictors to check the combining logic only; they say nothing about
model quality (that is what ``evaluate_hybrid.py`` measures).
"""

from __future__ import annotations

import importlib.util
from types import SimpleNamespace

import numpy as np
import pytest

from src.ml.char_tokenizer import OOV_ID, PAD_ID, CharTokenizer, clean_url
from src.ml.fusion import (
    FusionConfig, FusionConfigError, fuse, load_fusion_config, save_fusion_config, select_weights,
)
from src.ml.hybrid_predictor import HybridPredictor
from src.ml.lstm_config import LSTM_MODEL_PATH, LSTM_TOKENIZER_PATH
from src.ml.lstm_predictor import LSTMNotAvailableError, LSTMPredictor

HAS_TF = importlib.util.find_spec("tensorflow") is not None
requires_lstm = pytest.mark.skipif(
    not (HAS_TF and LSTM_MODEL_PATH.exists() and LSTM_TOKENIZER_PATH.exists()),
    reason="LSTM not trained / TensorFlow missing; run `python src/ml/train_lstm.py`.",
)

URLS = [
    "https://www.google.com/search?q=phishing",
    "http://secure-paypal.verify-account-login.tk/webscr/update.php?cmd=login",
    "",
    "http://a.com/" + "x" * 5000,
    "http://exämple.com/päth?q=%E2%9C%93&x=<script>alert(1)</script>#frag@@",
]


# ---------------- preprocessing ----------------

def test_scheme_is_stripped_consistently():
    assert clean_url("HTTPS://Example.com/A") == "example.com/a"
    assert clean_url("example.com/a") == "example.com/a"


@pytest.mark.parametrize("url", URLS + [None, 123])
def test_encoding_has_fixed_length_for_any_input(url):
    tok = CharTokenizer.fit(["abc.com/x", "def.org"], max_len=50)
    ids = tok.encode(url)
    assert len(ids) == 50 and all(isinstance(i, int) for i in ids)


def test_padding_truncation_and_oov():
    tok = CharTokenizer.fit(["ab"], max_len=4)
    assert tok.encode("a") == [tok.char_to_id["a"], PAD_ID, PAD_ID, PAD_ID]
    assert tok.encode("abababab") == [tok.char_to_id[c] for c in "abab"]
    assert tok.encode("z")[0] == OOV_ID
    assert tok.encode_batch([]).shape == (0, 4)


def test_tokenizer_roundtrip_gives_identical_encoding(tmp_path):
    tok = CharTokenizer.fit(URLS, max_len=120)
    tok.save(tmp_path / "tok.json")
    loaded = CharTokenizer.load(tmp_path / "tok.json")
    for url in URLS:
        assert loaded.encode(url) == tok.encode(url)


def test_vocabulary_is_deterministic():
    a = CharTokenizer.fit(URLS, max_len=10)
    b = CharTokenizer.fit(list(reversed(URLS)), max_len=10)
    assert a.char_to_id == b.char_to_id


# ---------------- fusion ----------------

def test_fuse_is_the_weighted_sum():
    cfg = FusionConfig(0.3, 0.7)
    assert fuse(0.2, 0.9, cfg) == pytest.approx(0.3 * 0.2 + 0.7 * 0.9)
    assert np.allclose(fuse([0.0, 1.0], [1.0, 0.0], cfg), [0.7, 0.3])


@pytest.mark.parametrize("rf,lstm", [(0.5, 0.6), (-0.1, 1.1), (0.2, 0.2)])
def test_invalid_weights_rejected(rf, lstm):
    with pytest.raises(FusionConfigError):
        FusionConfig(rf, lstm)


def test_select_weights_evaluates_the_grid_and_picks_the_best():
    y = np.array([0, 0, 1, 1] * 10)
    rf = np.where(y == 1, 0.9, 0.1)            # RF perfect
    lstm = np.where(y == 1, 0.4, 0.6)          # LSTM perfectly wrong
    cfg = select_weights(y, rf, lstm, metric="f1")
    assert len(cfg.selection["grid"]) == 11
    # Fused scores separate the classes only if rf_weight > 0.2 here, so the
    # selected weight must be in the perfect-F1 region and never LSTM-only.
    assert cfg.rf_weight > 0.2 and cfg.selection["best_validation_score"] == 1.0
    assert cfg.selection["best_validation_score"] == max(r["f1"] for r in cfg.selection["grid"])


def test_fusion_config_roundtrip_and_missing(tmp_path):
    save_fusion_config(FusionConfig(0.4, 0.6), tmp_path / "f.json")
    assert load_fusion_config(tmp_path / "f.json").lstm_weight == pytest.approx(0.6)
    with pytest.raises(FusionConfigError):
        load_fusion_config(tmp_path / "missing.json")


# ---------------- missing artefacts ----------------

def test_missing_lstm_artefacts_raise_a_clear_error(tmp_path):
    predictor = LSTMPredictor(tmp_path / "no.keras", tmp_path / "no.json")
    assert predictor.is_ready is False
    with pytest.raises(LSTMNotAvailableError, match="trained first"):
        predictor.predict("https://example.com")


def test_missing_tokenizer_alone_is_reported(tmp_path):
    (tmp_path / "m.keras").write_bytes(b"x")
    with pytest.raises(LSTMNotAvailableError):
        LSTMPredictor(tmp_path / "m.keras", tmp_path / "absent.json").predict("a.com")


# ---------------- hybrid plumbing (stand-in branches) ----------------

class _StubRF:
    suspicious_threshold, phishing_threshold = 0.35, 0.70

    def __init__(self, p):
        self.p = p

    def predict(self, url):
        return SimpleNamespace(phishing_probability=self.p, contributions=[{"feature": "x"}],
                               indicators=[{"level": "good", "text": "t"}])


class _StubLSTM:
    def __init__(self, p):
        self.p = p

    def predict_proba(self, url):
        return self.p


def test_hybrid_combines_both_probabilities():
    h = HybridPredictor(_StubRF(0.2), _StubLSTM(0.9), FusionConfig(0.25, 0.75))
    out = h.predict("http://x.tk/login")
    assert out["rf_probability"] == pytest.approx(0.2)
    assert out["lstm_probability"] == pytest.approx(0.9)
    assert out["probability"] == pytest.approx(0.25 * 0.2 + 0.75 * 0.9)
    assert out["prediction"] == 1 and out["label"] == "Phishing"
    assert out["rf_explanation"]["contributions"] == [{"feature": "x"}]


def test_hybrid_legitimate_case():
    out = HybridPredictor(_StubRF(0.05), _StubLSTM(0.1), FusionConfig(0.5, 0.5)).predict("a.com")
    assert out["prediction"] == 0 and out["label"] == "Legitimate" and out["verdict"] == "Safe"


def test_hybrid_propagates_missing_lstm(tmp_path):
    h = HybridPredictor(_StubRF(0.5), LSTMPredictor(tmp_path / "a", tmp_path / "b"), FusionConfig(0.5, 0.5))
    with pytest.raises(LSTMNotAvailableError):
        h.predict("a.com")


def test_hybrid_without_fusion_config_fails_loudly(monkeypatch, tmp_path):
    monkeypatch.setattr("src.ml.hybrid_predictor.load_fusion_config",
                        lambda: load_fusion_config(tmp_path / "none.json"))
    with pytest.raises(FusionConfigError):
        HybridPredictor(_StubRF(0.5), _StubLSTM(0.5)).predict("a.com")


# ---------------- real trained LSTM (skipped until trained) ----------------

@requires_lstm
@pytest.mark.parametrize("url", URLS)
def test_trained_lstm_returns_a_valid_probability(url):
    out = LSTMPredictor().predict(url)
    assert 0.0 <= out["probability"] <= 1.0 and out["prediction"] in (0, 1)


@requires_lstm
def test_trained_lstm_is_deterministic():
    p = LSTMPredictor()
    assert p.predict_proba("http://a.tk/login") == p.predict_proba("http://a.tk/login")
