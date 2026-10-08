"""Character-level tokenizer for raw URLs (no TensorFlow dependency).

The same object, serialised to JSON at training time, is reloaded at
prediction time, so training and inference share one preprocessing path.

Preprocessing
-------------
1. A leading ``scheme://`` is removed. The training corpus stores URLs
   without a scheme, while the web app normalises user input to include one;
   without this step every live URL would start with characters the LSTM never
   saw in training.
2. Optional lower-casing (on by default).
3. Each character is mapped to an integer id (``0`` = padding, ``1`` =
   out-of-vocabulary).
4. Sequences are truncated to ``max_len`` (keeping the *start*, i.e. the host)
   and right-padded with zeros.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import numpy as np

PAD_ID = 0
OOV_ID = 1
_SCHEME_RE = re.compile(r"^[a-z][a-z0-9+.\-]*://", re.IGNORECASE)
SCHEMA_VERSION = 1


def clean_url(url: Any, lowercase: bool = True) -> str:
    """Return the canonical text the LSTM sees (never raises)."""
    text = "" if url is None else str(url).strip()
    text = _SCHEME_RE.sub("", text, count=1)
    return text.lower() if lowercase else text


class CharTokenizer:
    """Maps URL strings onto fixed-length integer arrays."""

    def __init__(self, char_to_id: Dict[str, int], max_len: int, lowercase: bool = True) -> None:
        if max_len < 1:
            raise ValueError("max_len must be positive")
        self.char_to_id = dict(char_to_id)
        self.max_len = int(max_len)
        self.lowercase = bool(lowercase)

    # -- construction ------------------------------------------------------

    @classmethod
    def fit(
        cls, urls: Iterable[str], max_len: int, lowercase: bool = True, min_count: int = 1
    ) -> "CharTokenizer":
        """Build a vocabulary from training URLs only."""
        counts: Counter[str] = Counter()
        for url in urls:
            counts.update(clean_url(url, lowercase))
        # Sort for a deterministic id assignment: by frequency, then character.
        chars = sorted((c for c, n in counts.items() if n >= min_count),
                       key=lambda c: (-counts[c], c))
        mapping = {char: index + 2 for index, char in enumerate(chars)}
        return cls(mapping, max_len, lowercase)

    @property
    def vocab_size(self) -> int:
        """Embedding input size (characters + PAD + OOV)."""
        return len(self.char_to_id) + 2

    # -- encoding ----------------------------------------------------------

    def encode(self, url: Any) -> List[int]:
        """Encode one URL into exactly ``max_len`` ids."""
        text = clean_url(url, self.lowercase)[: self.max_len]
        ids = [self.char_to_id.get(char, OOV_ID) for char in text]
        ids.extend([PAD_ID] * (self.max_len - len(ids)))
        return ids

    def encode_batch(self, urls: Sequence[Any]) -> np.ndarray:
        """Encode many URLs into an ``(n, max_len)`` int32 array."""
        if len(urls) == 0:
            return np.zeros((0, self.max_len), dtype=np.int32)
        return np.asarray([self.encode(url) for url in urls], dtype=np.int32)

    # -- persistence ---------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "max_len": self.max_len,
            "lowercase": self.lowercase,
            "char_to_id": self.char_to_id,
        }

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "CharTokenizer":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if data.get("schema_version") != SCHEMA_VERSION:
            raise ValueError("Unsupported tokenizer schema version.")
        return cls(data["char_to_id"], data["max_len"], data.get("lowercase", True))
