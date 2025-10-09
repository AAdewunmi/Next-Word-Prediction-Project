````markdown
# Next-Word Prediction (Django + Keras)

A Django web app that serves a word-level next-word predictor trained from **Predict_words.ipynb**.
Users provide exactly _N_ tokens (default: 50), the app normalizes input like training,
and returns the continuation predicted by the model.

## Features

- Clean UI with a single text box and inline validation (exact token count)
- Server-side preprocessing consistent with the notebook
- Fast single-pass greedy inference (sampling is easy to add)
- DB logging of requests/responses for analytics
- Tests (pytest + pytest-django) and CI (GitHub Actions)

---

## Setup

### 1) Clone & Python env

```bash
git clone https://github.com/AAdewunmi/Next-Word-Prediction-Project.git
cd Next-Word-Prediction-Project/nextword_site
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -U pip
pip install -r requirements.txt
```
````

### 2) Place model artifacts

Put your trained files under the default app directory (or set env vars below):

```
predictor/artifacts/
├── metadata.json                  # optional but recommended (contains {"seq_length": 50, ...})
├── tokenizer.pkl
└── nextWordPredict/
    └── nextWord.h5
```

If you prefer custom locations, set in `.env`:

```env
ARTIFACTS_DIR=/abs/path/to/artifacts
MODEL_PATH=/abs/path/to/nextWordPredict/nextWord.h5
TOKENIZER_PATH=/abs/path/to/tokenizer.pkl
METADATA_PATH=/abs/path/to/metadata.json
# SEQ_LENGTH=50   # optional override if metadata/model not available
```

### 3) Django app

```bash
cp .env.example .env
python manage.py migrate
python manage.py runserver
```

Open [http://127.0.0.1:8000/](http://127.0.0.1:8000/) and try a prompt with **exactly N tokens** (displayed at the top of the page).

---

## Using Google Drive / Colab to Produce Artifacts

1. In Colab:

   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   ```

2. Train your model in `Predict_words.ipynb`.
3. Save artifacts (model `.keras`, `tokenizer.pkl`, and `metadata.json`) to Drive.
4. Download them or sync to your repo under `predictor/artifacts/` (or point env vars to your Drive mount if deploying on a VM with Drive mounted).

> If you used the notebook template we discussed, your metadata file already includes `seq_length`. The Django app reads it automatically; no hard-coding needed.

---

## Testing

```bash
pytest -q
```

Tests stub the TensorFlow predictor so CI runs fast and green without a GPU/AVX-capable runner.

---

## Operations

- **Change context length**: update `metadata.json` (`seq_length`) or set `SEQ_LENGTH` in `.env`.
- **Database**: defaults to SQLite. For Postgres set `DATABASE_URL` in `.env`:

  ```
  DATABASE_URL=postgres://USER:PASS@HOST:5432/DBNAME
  ```

- **Static files**: `STATIC_ROOT=staticfiles` (set up WhiteNoise if deploying as pure WSGI).

---

## Notes

- Output style depends strongly on training choices. Greedy decoding tends to repeat; switch to sampling if you want more variety.
- If you see “tokenizer vocab exceeds model embedding input_dim”, you loaded mismatched artifacts. Use the _matching_ pair from the same training run.

## License

MIT (or your choice).

````

---

## Runbook — quick commands

```bash
# install
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# env + db + run
cp .env.example .env
python manage.py migrate
python manage.py runserver

# tests
pytest -q
````

---

### Final notes

- The form enforces **exact** token count pre-submit; the service enforces it **after normalization**, mirroring training. That double-gate prevents silent mismatches.
- The predictor is a singleton and lazy-loads artifacts. If you want a warm-up at boot, add a Django `AppConfig.ready()` hook to call `get_predictor().get_seq_length()`.
- If you want sampling (temperature/top-k) instead of greedy, I can drop that into `services/predictor.py` as `predict_sampled(...)` and toggle via an env flag.

---

### License

[MIT LICENSE](LICENSE)
