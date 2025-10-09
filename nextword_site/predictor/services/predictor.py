import json
import pickle
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from django.conf import settings
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

from ..utils.text import normalize_text_ascii_letters_only
from tensorflow.keras.models import load_model

@dataclass
class ArtifactsConfig:
    """Configuration for model/tokenizer/metadata locations."""
    model_path: Path
    tokenizer_path: Path
    metadata_path: Path
    seq_length_override: Optional[int] = None


def _read_metadata_seq_length(path: Path) -> Optional[int]:
    if not path.exists():
        return None
    try:
        meta = json.loads(path.read_text(encoding="utf-8"))
        sl = int(meta.get("seq_length")) if meta.get("seq_length") is not None else None
        return sl if sl and sl > 0 else None
    except Exception:
        return None


def _infer_seq_length_from_model(model) -> Optional[int]:
    try:
        shape = getattr(model, "input_shape", None)
        if isinstance(shape, (list, tuple)) and len(shape) >= 2 and isinstance(shape[1], int):
            return int(shape[1])
    except Exception:
        pass
    return None


class NextWordPredictor:
    """
    Load the trained next-word model + tokenizer and perform inference.
    Model/tokenizer are loaded once and cached for subsequent requests.
    """
    def __init__(self, cfg: ArtifactsConfig):
        self.cfg = cfg
        self._model = None
        self._tokenizer = None
        self._seq_length_cached: Optional[int] = None

    def _load(self):
        if self._model is None:
            p = self.cfg.model_path
            try:
                self._model = load_model(p, compile=False)
            except Exception:
                # fallback: swap suffix between .h5 <-> .keras if present
                alt = p.with_suffix(".keras") if p.suffix == ".h5" else p.with_suffix(".h5")
                self._model = load_model(alt, compile=False)
    # tokenizer as above...


    def get_seq_length(self) -> int:
        """
        Determine the required sequence length:
        1) explicit override from settings/env
        2) metadata.json
        3) model.input_shape
        4) default 50
        """
        if self._seq_length_cached:
            return self._seq_length_cached

        if self.cfg.seq_length_override:
            self._seq_length_cached = self.cfg.seq_length_override
            return self._seq_length_cached

        # Try metadata
        sl = _read_metadata_seq_length(self.cfg.metadata_path)
        if sl:
            self._seq_length_cached = sl
            return sl

        # Load model solely to ask for input shape
        self._load()
        sl = _infer_seq_length_from_model(self._model)
        self._seq_length_cached = sl if sl else 50
        return self._seq_length_cached

    def predict(self, seed_text: str, n_words: int = 12) -> Tuple[str, int]:
        """
        Run greedy next-word generation. Returns (generated_text, latency_ms).
        """
        self._load()
        seq_length = self.get_seq_length()

        # Preprocess like training
        cleaned = normalize_text_ascii_letters_only(seed_text)

        # Ensure seed is exactly seq_length tokens after cleaning
        tokens = cleaned.split()
        if len(tokens) != seq_length:
            raise ValueError(f"Expected {seq_length} tokens after normalization; got {len(tokens)}.")

        in_text = " ".join(tokens)

        idx_to_word = getattr(self._tokenizer, "index_word", {})
        start = time.perf_counter()

        for _ in range(n_words):
            enc = self._tokenizer.texts_to_sequences([in_text])[0]
            enc = pad_sequences([enc], maxlen=seq_length, truncating="pre")
            probs = self._model.predict(enc, verbose=0)[0]
            next_id = int(np.argmax(probs))
            word = idx_to_word.get(next_id)
            if not word:
                break
            in_text += " " + word

        latency_ms = int((time.perf_counter() - start) * 1000)
        return in_text, latency_ms


# --- Singleton accessor used by forms/views ---
_predictor_singleton: Optional[NextWordPredictor] = None


def get_predictor() -> NextWordPredictor:
    global _predictor_singleton
    if _predictor_singleton is None:
        cfg = ArtifactsConfig(
            model_path=settings.MODEL_PATH,
            tokenizer_path=settings.TOKENIZER_PATH,
            metadata_path=settings.METADATA_PATH,
            seq_length_override=settings.SEQ_LENGTH,
        )
        _predictor_singleton = NextWordPredictor(cfg)
    return _predictor_singleton


def get_required_seq_length() -> int:
    """
    Convenience used by the form to validate token count without loading TF too early:
    - Prefer metadata.json or override; only loads model if needed.
    """
    return get_predictor().get_seq_length()
