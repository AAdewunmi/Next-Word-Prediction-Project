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
            self._model = load_model(self.cfg.model_path)
        if self._tokenizer is None:
            with self.cfg.tokenizer_path.open("rb") as f:
                self._tokenizer = pickle.load(f)