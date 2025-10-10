import json
import pickle
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from django.conf import settings
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.preprocessing.sequence import pad_sequences

from ..utils.text import normalize_text_ascii_letters_only


# --- Keras 3 → tf.keras tokenizer pickle compatibility ---
def _load_tokenizer_compat(pickle_path: Path):
    """
    Load a Tokenizer pickled in Keras 3 (module path 'keras.src...')
    into a TF 2.10 / Keras 2 runtime by aliasing modules, then (optionally)
    rehydrate to a native tf.keras Tokenizer via JSON.
    """
    import io
    import pickle
    import sys
    import types
    import tensorflow.keras as tfk

    raw = pickle_path.read_bytes()

    # First attempt: works if pickle was created with tf.keras already
    try:
        tok = pickle.loads(raw)
    except ModuleNotFoundError as e:
        if "keras.src" not in str(e):
            raise

        # Build module aliases so 'keras.src.*' resolves to tf.keras.*
        #  - 'keras' → a proxy module that exposes tf.keras attributes
        #  - 'keras.src' → tf.keras
        #  - 'keras.src.preprocessing' → tf.keras.preprocessing
        #  - 'keras.src.preprocessing.text' → tf.keras.preprocessing.text
        keras_proxy = sys.modules.get("keras")
        if not keras_proxy:
            keras_proxy = types.ModuleType("keras")
            # mirror tf.keras attributes for direct access (best-effort)
            keras_proxy.__dict__.update(tfk.__dict__)
            sys.modules["keras"] = keras_proxy

        sys.modules["keras.src"] = tfk
        try:
            sys.modules["keras.src.preprocessing"] = tfk.preprocessing
            sys.modules["keras.src.preprocessing.text"] = tfk.preprocessing.text
        except Exception:
            pass

        # Retry unpickle with the aliases in place
        tok = pickle.loads(raw)

    # Optional: normalize to a native tf.keras Tokenizer instance
    try:
        from tensorflow.keras.preprocessing.text import tokenizer_from_json
        tok = tokenizer_from_json(tok.to_json())
    except Exception:
        # Best-effort only; if to_json() unavailable, continue with the loaded object
        pass

    return tok


@dataclass
class ArtifactsConfig:
    """Configuration for model/tokenizer/metadata locations."""
    model_path: Path
    tokenizer_path: Path
    metadata_path: Path
    seq_length_override: Optional[int] = None


def _read_metadata_seq_length(path: Path) -> Optional[int]:
    """Read sequence length from metadata file."""
    if not path.exists():
        return None
    try:
        meta = json.loads(path.read_text(encoding="utf-8"))
        sl = int(meta.get("seq_length")) if meta.get("seq_length") is not None else None
        return sl if sl and sl > 0 else None
    except Exception:
        return None


def _infer_seq_length_from_model(model) -> Optional[int]:
    """Infer sequence length from model input shape."""
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
        """
        Load model/tokenizer once and cache.
        Strategy:
        1) Try to load full model (config+weights) via load_model(compile=False).
        2) If that fails due to Keras 2/3 config mismatch (e.g. 'batch_shape'),
            rebuild the architecture from metadata and load weights via load_weights().
        """
        # Always load tokenizer first so we can infer vocab_size if metadata is missing
        if self._tokenizer is None:
            with self.cfg.tokenizer_path.open("rb") as f:
                self._tokenizer = pickle.load(f)

        if self._model is None:
            p = Path(self.cfg.model_path)

            # Candidate paths (prefer .h5 for TF 2.10, then .keras)
            candidates: list[Path] = []
            if p.suffix in {".h5", ".keras"}:
                # Try as given, then the alternate suffix
                candidates.append(p)
                candidates.append(p.with_suffix(".keras" if p.suffix == ".h5" else ".h5"))
            else:
                # No suffix: try .h5 first (best for TF 2.10), then .keras
                candidates.extend([p.with_suffix(".h5"), p.with_suffix(".keras")])

            last_err: Optional[Exception] = None

            # 1) Try native load_model first
            for path in candidates:
                if not path.exists():
                    continue
                try:
                    self._model = load_model(path, compile=False)
                    self.cfg.model_path = path  # remember what worked
                    last_err = None
                    break
                except Exception as e:
                    # Save and try the rebuild route below
                    last_err = e
                    # Don’t break; we’ll attempt rebuild using the first existing candidate
                    self.cfg.model_path = path
                    break  # we rebuild using this path

            if self._model is None:
                if not self.cfg.model_path.exists():
                    raise FileNotFoundError(
                        "Model file not found. Tried:\n  " +
                        "\n  ".join(str(x) for x in candidates)
                    )

                # 2) Rebuild from metadata + load weights
                # Read metadata (dimensions used during training)
                meta = {}
                try:
                    if settings.METADATA_PATH and Path(settings.METADATA_PATH).exists():
                        meta = json.loads(Path(settings.METADATA_PATH).read_text(encoding="utf-8"))
                except Exception:
                    meta = {}

                # Required dims
                seq_len = (
                    settings.SEQ_LENGTH
                    or meta.get("seq_length")
                    or None
                )
                if not seq_len:
                    # Last resort: keep whatever get_seq_length() will infer later
                    # but we do need a concrete integer here:
                    maybe = self._seq_length_cached or 50
                    seq_len = int(maybe)

                # Vocab size: prefer metadata, else tokenizer
                vocab_size = meta.get("vocab_size") or (len(getattr(self._tokenizer, "word_index", {})) + 1)
                embedding_dim = int(meta.get("embedding_dim") or 50)
                lstm_units = int(meta.get("lstm_units") or 50)
                dense_units = int(meta.get("dense_units") or 50)

                # Build the architecture you used during training
                m = Sequential()
                m.add(Embedding(int(vocab_size), embedding_dim, input_length=int(seq_len)))
                m.add(LSTM(lstm_units, return_sequences=True))
                m.add(LSTM(lstm_units))
                m.add(Dense(dense_units, activation="relu"))
                m.add(Dense(int(vocab_size), activation="softmax"))
                m.build(input_shape=(None, int(seq_len)))

                # Load weights from the same .h5 (or .keras) file
                # This reads only the 'model_weights' group from the file.
                try:
                    m.load_weights(self.cfg.model_path)
                except Exception as e2:
                    raise RuntimeError(
                        f"Could not load weights from {self.cfg.model_path} after rebuild.\n"
                        f"Original load_model error: {last_err}\n"
                        f"Weights load error: {e2}"
                    ) from e2

                self._model = m
                # Ensure seq length cache is aligned
                self._seq_length_cached = int(seq_len)


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

        sl = _read_metadata_seq_length(self.cfg.metadata_path)
        if sl:
            self._seq_length_cached = sl
            return sl

        self._load()
        sl = _infer_seq_length_from_model(self._model)
        self._seq_length_cached = sl if sl else 50
        return self._seq_length_cached

    def predict(self, seed_text: str, n_words: int = 12) -> Tuple[str, int]:
        """Run greedy next-word generation. Returns (generated_text, latency_ms)."""
        self._load()
        seq_length = self.get_seq_length()

        cleaned = normalize_text_ascii_letters_only(seed_text)

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


# Singleton accessor
_predictor_singleton: Optional[NextWordPredictor] = None


def get_predictor() -> NextWordPredictor:
    """Get or create the predictor singleton instance."""
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
    Convenience used by the form to validate token count without loading TF too early.
    Prefer metadata.json or override; only loads model if needed.
    """
    return get_predictor().get_seq_length()
