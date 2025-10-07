# Predict-Words — Next-Word Generation (Colab + Google Drive)

A notebook-driven workflow for training and running a word-level LSTM next-word predictor on *The Republic* by Plato. This project is designed to run **in Google Colab** with all data/models persisted to **Google Drive** so your work survives Colab sessions.

---

## Contents

* [What you get](#what-you-get)
* [Folder layout in Google Drive](#folder-layout-in-google-drive)
* [Quick start (Colab)](#quick-start-colab)
* [Set up Google Drive + Colab](#set-up-google-drive--colab)
* [Data prep & tokenization](#data-prep--tokenization)
* [Train, save, and reload the model](#train-save-and-reload-the-model)
* [Generate text (greedy or sampling)](#generate-text-greedy-or-sampling)
* [Utilities: Drive-backed I/O](#utilities-drive-backed-io)
* [Troubleshooting](#troubleshooting)
* [FAQ](#faq)

---

## What you get

* **Notebook**: `Predict_words.ipynb` (job: prepare data → train LSTM → save artifacts → generate text).
* **Drive-backed persistence**: all `.txt`, `.pkl`, and `.keras` files are saved under a project folder in **MyDrive**.
* **Utilities**: helper functions for Drive path resolution, document I/O, social-text cleaning (optional), training/persistence, and generation.
* **Reproducible paths**: consistent with Colab’s `/content/drive/MyDrive/...`.

---

## Folder layout in Google Drive

Create this structure in **Google Drive**:

```
MyDrive/
└── Colab Notebooks/
    └── Predict-Words-Analysis/
        ├── data/                   # *.txt corpora & prepared sequence files
        │   ├── republic.txt
        │   └── republic_sequences.txt
        └── models/
            ├── tokenizer.pkl       # Keras Tokenizer (pickle)
            ├── metadata.json       # saved by the training cell
            └── nextWordPredict/
                └── nextWord.keras  # trained model (Keras v3 format)
```

> Note the exact capitalization and the space in **“Colab Notebooks”** and **“MyDrive”**.

---

## Quick start (Colab)

1. **Open Colab** and mount Drive:

   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   ```

2. **Put the notebook** `Predict_words.ipynb` anywhere in Drive (e.g., in the project folder above) and open it in Colab.

3. **Place data**
   Copy `republic.txt` into:

   ```
   /content/drive/MyDrive/Colab Notebooks/Predict-Words-Analysis/data/
   ```

4. **Run the cells in order** (see sections below).
   The notebook will create `republic_sequences.txt`, train a model, and save:

   * `models/nextWordPredict/nextWord.keras`
   * `models/tokenizer.pkl`
   * `models/metadata.json`

5. **Generate text** by loading the saved artifacts from Drive.

---

## Set up Google Drive + Colab

Add this to the **first cell** of the notebook:

```python
from google.colab import drive
drive.mount('/content/drive')
```

Define project paths (used by helper functions and cells that follow):

```python
from pathlib import Path

DRIVE_BASE   = Path("/content/drive/MyDrive")
PROJECT_ROOT = DRIVE_BASE / "Colab Notebooks" / "Predict-Words-Analysis"
DATA_DIR     = PROJECT_ROOT / "data"
MODELS_DIR   = PROJECT_ROOT / "models"
NEXTWORD_DIR = MODELS_DIR / "nextWordPredict"

for d in (DATA_DIR, MODELS_DIR, NEXTWORD_DIR):
    d.mkdir(parents=True, exist_ok=True)
```

---

## Data prep & tokenization

### 1) Save/Load documents (Drive-aware)

```python
from pathlib import Path
from typing import List, Union
import string

def _resolve_path(filename: Union[str, Path]) -> Path:
    p = Path(filename)
    if p.is_absolute():
        return p
    # route by extension
    if p.suffix.lower() == ".txt":
        return DATA_DIR / p.name
    return PROJECT_ROOT / p.name

def load_doc(filename: str) -> str:
    path = _resolve_path(filename)
    return path.read_text(encoding="utf-8")

def save_doc(lines: List[str], filename: str) -> None:
    path = _resolve_path(filename)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
```

### 2) Clean + sequence preparation

If you already have `republic.txt`, build sequences for next-word prediction (classic “`seq_length + 1` per line`” format):

```python
text = load_doc("republic.txt").lower()

# very simple cleaner (ASCII letters only). Adjust if you want punctuation.
import re
text = re.sub(r"[^a-z\s]", " ", text)
text = re.sub(r"\s+", " ", text).strip()

# build sequences
words = text.split()
seq_length = 50  # adjust (longer context usually helps)
sequences = []
for i in range(seq_length, len(words)):
    seq = words[i-seq_length:i+1]        # seq_length context + 1 target
    sequences.append(" ".join(seq))

save_doc(sequences, "republic_sequences.txt")
print("Sequences:", len(sequences))
```

---

## Train, save, and reload the model

### 1) Vectorize + model training (single cell)

```python
import numpy as np
from numpy import array
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Embedding
import json, pickle, time, platform

# Load sequences from Drive
lines = load_doc("republic_sequences.txt").splitlines()
tokenizer = Tokenizer()
tokenizer.fit_on_texts(lines)
encoded = tokenizer.texts_to_sequences(lines)

vocab_size = len(tokenizer.word_index) + 1
encoded = array(encoded)
X, y = encoded[:, :-1], encoded[:, -1]
seq_length = X.shape[1]
y = to_categorical(y, num_classes=vocab_size)

# Define model
embedding_dim = 50
lstm_units = 50
dense_units = 50

model = Sequential([
    Embedding(vocab_size, embedding_dim, input_length=seq_length),
    LSTM(lstm_units, return_sequences=True),
    LSTM(lstm_units),
    Dense(dense_units, activation='relu'),
    Dense(vocab_size, activation='softmax'),
])
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())

# Train
history = model.fit(X, y, batch_size=128, epochs=50)

# Persist artifacts to Drive
NEXTWORD_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

model_path     = NEXTWORD_DIR / "nextWord.keras"
tokenizer_path = MODELS_DIR / "tokenizer.pkl"
metadata_path  = MODELS_DIR / "metadata.json"

model.save(model_path)
with tokenizer_path.open("wb") as f:
    pickle.dump(tokenizer, f)

meta = {
    "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "seq_length": int(seq_length),
    "vocab_size": int(vocab_size),
    "embedding_dim": int(embedding_dim),
    "lstm_units": int(lstm_units),
    "dense_units": int(dense_units),
    "python_version": platform.python_version(),
}
metadata_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

print("Saved:", model_path, tokenizer_path, metadata_path)
```

### 2) Reload for inference

```python
from tensorflow.keras.models import load_model

model = load_model(NEXTWORD_DIR / "nextWord.keras")
with (MODELS_DIR / "tokenizer.pkl").open("rb") as f:
    tokenizer = pickle.load(f)

print("Loaded model & tokenizer.")
```

---

## Generate text (greedy or sampling)

**Greedy** (simple, tends to repeat):

```python
from tensorflow.keras.preprocessing.sequence import pad_sequences

def infer_seq_length_from_model(model) -> int:
    shape = getattr(model, "input_shape", None)
    return int(shape[1]) if isinstance(shape, (list, tuple)) and isinstance(shape[1], int) else seq_length

def generate_greedy(model, tokenizer, seed_text: str, n_words: int = 20) -> str:
    seq_len = infer_seq_length_from_model(model)
    in_text = seed_text.strip()
    idx2word = getattr(tokenizer, "index_word", {})
    for _ in range(n_words):
        encoded = tokenizer.texts_to_sequences([in_text])[0]
        encoded = pad_sequences([encoded], maxlen=seq_len, truncating="pre")
        yhat = model.predict(encoded, verbose=0).argmax(axis=-1)[0]
        word = idx2word.get(yhat)
        if not word:
            break
        in_text += " " + word
    return in_text

# Pick a seed from your sequences
seed = lines[np.random.randint(len(lines))]
seed = " ".join(seed.split()[:seq_length])
print(generate_greedy(model, tokenizer, seed, n_words=12))
```

**Sampling** (readable, less repetitive):

```python
import numpy as np

def generate_sampling(model, tokenizer, seed_text: str, n_words=30, temperature=0.9, top_k=50):
    seq_len = infer_seq_length_from_model(model)
    idx2word = getattr(tokenizer, "index_word", {})
    in_text = seed_text.strip()
    for _ in range(n_words):
        enc = tokenizer.texts_to_sequences([in_text])[0]
        enc = pad_sequences([enc], maxlen=seq_len, truncating="pre")
        probs = model.predict(enc, verbose=0)[0]

        # temperature
        logits = np.log(probs + 1e-9) / max(temperature, 1e-6)
        probs_t = np.exp(logits); probs_t /= probs_t.sum()

        # top-k
        if top_k:
            idxs = np.argpartition(probs_t, -top_k)[-top_k:]
            p = probs_t[idxs] / probs_t[idxs].sum()
            next_id = int(np.random.choice(idxs, p=p))
        else:
            next_id = int(np.random.choice(len(probs_t), p=probs_t))

        word = idx2word.get(next_id)
        if not word:
            break
        in_text += " " + word
    return in_text

print(generate_sampling(model, tokenizer, seed, n_words=30, temperature=0.9, top_k=50))
```

---

## Utilities: Drive-backed I/O

If you prefer, keep a small helper to route relative filenames into the project folders automatically:

```python
def project_path(filename: str) -> Path:
    p = Path(filename)
    if p.is_absolute():
        return p
    if p.suffix.lower() == ".txt":
        return DATA_DIR / p.name
    if p.suffix.lower() == ".pkl":
        return MODELS_DIR / p.name
    if p.suffix.lower() in (".keras", ".h5"):
        return NEXTWORD_DIR / p.name
    return PROJECT_ROOT / p.name
```

---

## Troubleshooting

**`FileNotFoundError: 'Mydrive/...'`**
Use the correct Colab path prefix and capitalization:

```
/content/drive/MyDrive/Colab Notebooks/Predict-Words-Analysis/...
```

Mount Drive first.

**Tokenizer/model mismatch**
If generation is nonsense or you see:

```
Tokenizer vocab (...) exceeds model embedding input_dim (...)
```

you likely loaded a tokenizer from a different training run. Always use the **matching** `tokenizer.pkl` for a given `nextWord.keras`.

**Training logs show `1/1 ...` many times**
That’s Keras `predict` verbosity. In generation code, pass `verbose=0`.

**Output reads like “they they not not”**
Greedy decoding on a small word-level model will do that. Use the **sampling** generator (temperature + top-k), keep punctuation during training, or increase model size/context.

**Spaces in folder names**
`Colab Notebooks` has a space. Using `Path` as shown handles this correctly.

---

## FAQ

**Can I save as `.h5` instead of `.keras`?**
Yes:

```python
model.save(NEXTWORD_DIR / "nextWord.h5")
```

`.keras` is the recommended modern format.

**How do I train on a different text?**
Drop your corpus as `data/your_text.txt`, rebuild sequences, then rerun the training cell. Keep tokenizer+model pairs together.

**How do I persist across sessions?**
Always save under **MyDrive** (as shown). Anything under `/content/...` alone will be ephemeral.

---

### License

[MIT LICENSE](LICENSE)
