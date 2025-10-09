import json
from pathlib import Path
import pytest
from predictor.services.predictor import ArtifactsConfig, NextWordPredictor


class DummyModel:
    input_shape = (None, 4)
    
    def predict(self, arr, verbose=0):
        import numpy as np
        p = np.zeros((1, 10)); p[0, 1] = 1.0
        return p


class DummyTok:
    index_word = {1: "next"}
    def texts_to_sequences(self, xs): return [[1, 2, 3, 4]]


@pytest.fixture
def tmp_artifacts(tmp_path: Path):
    (tmp_path / "nextWordPredict").mkdir(parents=True)
    (tmp_path / "tokenizer.pkl").write_bytes(b"\x80\x04.")  # dummy pickle to bypass existence checks
    (tmp_path / "nextWordPredict" / "nextWord.h5").write_bytes(b"stub")
    (tmp_path / "metadata.json").write_text(json.dumps({"seq_length": 4}))
    return tmp_path


def test_seq_length_resolution(monkeypatch, tmp_artifacts: Path):
    cfg = ArtifactsConfig(
        model_path=tmp_artifacts / "nextWordPredict" / "nextWord.h5",
        tokenizer_path=tmp_artifacts / "tokenizer.pkl",
        metadata_path=tmp_artifacts / "metadata.json",
        seq_length_override=None,
    )
    p = NextWordPredictor(cfg)
    # patch loaders
    monkeypatch.setattr(p, "_load", lambda: None)
    p._model = DummyModel()
    p._tokenizer = DummyTok()
    assert p.get_seq_length() == 4
