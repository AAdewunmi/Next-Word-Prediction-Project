# Next-Word Prediction — Django App

A Django web app that serves a word-level LSTM next-word predictor trained from a Colab notebook. Users submit **exactly N words** (default **50**) and get a continuation from the trained model. The app normalizes input the same way the notebook did (lowercase, ASCII letters only, collapse spaces), validates token count, and logs predictions.

> **Artifacts expected**:
>
> * `predictor/artifacts/nextWordPredict/nextWord.h5` (model weights; TF 2.10 friendly)
> * `predictor/artifacts/tokenizer.pkl` **or** `predictor/artifacts/tokenizer.json` (tokenizer; JSON preferred)
> * `predictor/artifacts/metadata.json` (contains at least `"seq_length"`)
> * `predictor/artifacts/exactly-50-word-seed.txt` (handy seed text for testing)

---

## Features

* **UX**

  * Single **Seed Text** textarea with inline validation (must be exactly `seq_length` words).
  * **Predict** button and **Clear** button (clears seed + prediction on the page).
  * Explanation panel showing the required token count and normalization behavior.
* **Inference**

  * Loads model and tokenizer once (cached).
  * **Decoding strategy**: sampling with temperature + top-k/top-p and a light repetition penalty (greedy available).
  * Normalization mirrors training: lowercase → letters only → collapse spaces.
* **Resilience**

  * Accepts **`.h5`** model (TF 2.10) and will **rebuild architecture + load weights** if a newer save format confuses `load_model`.
  * Tokenizer **pickle compat shim** for Keras 3 → tf.keras 2; prefers **`tokenizer.json`** when present.
* **Data**

  * `PredictionLog` table stores request/response, timing, IP/user agent, success/error, and timestamps.
* **DevOps**

  * `pytest` test suite (forms, views, services).
  * GitHub Actions CI workflow.
  * Platform-aware `requirements.txt` (Intel macOS vs Apple Silicon vs Linux).

---

## Screenshots

> Home (empty)
> 
> <img width="955" height="552" alt="Image" src="https://github.com/user-attachments/assets/2818af2d-9ccd-4e5c-bdfb-1475519b6747" />
>
> Prediction result
> 
> <img width="948" height="760" alt="Image" src="https://github.com/user-attachments/assets/8464b225-d301-4137-bbdc-69215dcc1c8e" />

---

## Requirements

* **Python**:

  * Intel macOS: **3.10** (recommended for TF 2.10)
  * Apple Silicon: 3.11 works (use `tensorflow-macos`)
  * Linux/Windows: 3.10–3.11 with standard `tensorflow`
* **TensorFlow**:

  * Intel macOS: **2.10.1**
  * Apple Silicon: 2.16.1 (`tensorflow-macos`)
  * Linux/Windows: 2.16.1 (`tensorflow`)

We’ve already pinned these in `requirements.txt` using PEP 508 markers.

---

## Project layout

```
nextword_site/
├─ manage.py
├─ .env.example
├─ requirements.txt
├─ .github/workflows/ci.yml
├─ nextword_site/
│  ├─ settings.py
│  ├─ urls.py
│  └─ wsgi.py
└─ predictor/
   ├─ apps.py, admin.py, urls.py, forms.py, models.py, views.py
   ├─ templates/predictor/home.html
   ├─ services/predictor.py
   ├─ utils/text.py
   ├─ tests/...
   └─ artifacts/
      ├─ metadata.json
      ├─ tokenizer.pkl            # or tokenizer.json (preferred)
      └─ nextWordPredict/
         └─ nextWord.h5
```

---

## Setup

### 1) Virtualenv + dependencies

```bash
# macOS Intel recommended flow
pyenv install 3.10.14   # if not already available
pyenv local 3.10.14
python -m venv .venv310
source .venv310/bin/activate

python -m pip install -U pip
python -m pip install -r requirements.txt
```

### 2) Configure environment

Copy the example file and edit paths:

```bash
cp .env.example .env
```

`.env` keys you’ll likely set:

```ini
SECRET_KEY=change-me
DEBUG=true
ALLOWED_HOSTS=localhost,127.0.0.1

# Absolute paths to artifacts on your machine
MODEL_PATH=/abs/path/to/nextword_site/predictor/artifacts/nextWordPredict/nextWord.h5
TOKENIZER_PATH=/abs/path/to/nextword_site/predictor/artifacts/tokenizer.json
# or TOKENIZER_PATH=/abs/path/to/nextword_site/predictor/artifacts/tokenizer.pkl
METADATA_PATH=/abs/path/to/nextword_site/predictor/artifacts/metadata.json

# Decoding (optional; defaults are sensible)
DECODE_STRATEGY=sampling         # or greedy
TEMPERATURE=0.9
TOP_K=50                         # set 0 to disable
TOP_P=0.0                        # e.g., 0.9 for nucleus sampling
REPETITION_PENALTY=1.15
RECENT_WINDOW=20

# Database (optional; defaults to SQLite)
# DATABASE_URL=postgres://user:pass@localhost:5432/nextword
```

### 3) Database

```bash
python manage.py migrate
```

### 4) Run

```bash
export TF_CPP_MIN_LOG_LEVEL=2  # optional: quiet TF logs
python manage.py runserver
# http://127.0.0.1:8000/
```

---

## Artifacts (what to place where)

* **Model weights**: `predictor/artifacts/nextWordPredict/nextWord.h5`
  Saved from Colab with `model.save(".../nextWord.h5")`
* **Tokenizer**:

  * Prefer `predictor/artifacts/tokenizer.json` (write in Colab with `tokenizer.to_json()`).
  * If you only have a pickle (`tokenizer.pkl`), keep it — the server has a compat loader for Keras-3 pickles.
* **Metadata**: `predictor/artifacts/metadata.json`
  At minimum `{"seq_length": 50}` so the app can validate tokens without guessing.
* **Seed text** (handy): `predictor/artifacts/exactly-50-word-seed.txt`
  Use this to copy/paste into the Seed Text box.

**Example seed file content (50 words, lowercase, letters-only):**

```
this simple seed text is designed to contain exactly fifty words for testing the next word prediction service in your django application ensuring that whitespace and casing do not change the token count during normalization and that the model can generate a continuation without raising errors or warnings today safely
```

> Count check: `len(open('.../exactly-50-word-seed.txt').read().split()) == 50`

---

## Usage

1. Open the home page.
2. Paste **exactly 50 words** (the app shows the required count) into **Seed Text**.
3. Click **Predict next words**.
4. Use **Clear** to reset the Seed and the Prediction textboxes without reloading.

If you want “more human” text, sampling is already the default. Tweak in `.env`:

```
DECODE_STRATEGY=sampling
TEMPERATURE=0.8
TOP_K=50
TOP_P=0.0
REPETITION_PENALTY=1.2
```

---

## How the pieces fit

* **Normalization** (server): lowercase → remove non-letters → collapse spaces (mirrors training).
* **Tokenizer**: loaded from JSON (preferred) or pickled with a Keras-3→tf.keras shim.
* **Model**:

  * First tries `load_model(..., compile=False)`.
  * If that fails (e.g., `batch_shape` from newer Keras), it **rebuilds** the architecture from `metadata.json` and loads weights via `load_weights(h5)`.
* **Decoding**: sampling (temp + top-k/top-p + repetition penalty) or greedy.
* **Logging**: `PredictionLog` captures inputs, outputs, timing, and request metadata.

---

## Testing

```bash
pytest -q
```

Tests stub the predictor for speed (no heavy TF in CI). Add more tests under `predictor/tests/`.

---

## CI (GitHub Actions)

The workflow at `.github/workflows/ci.yml` installs deps, runs Django checks, and executes tests on pushes and PRs. It relies on the platform-aware requirements to avoid TF headaches on Linux runners.

---

## Troubleshooting

* **“No file or directory found at … nextWord.keras”**
  We deploy **`nextWord.h5`** for TF 2.10. Update `MODEL_PATH` to point at `.h5` (or keep both; the loader tries both).
* **`ValueError: Unrecognized keyword arguments: ['batch_shape']`**
  H5 was saved by newer Keras. Loader auto-rebuilds the architecture and calls `load_weights`. No retrain needed.
* **`ModuleNotFoundError: No module named 'keras.src'` when loading tokenizer**
  That’s a Keras 3 pickle. We shim it; or export `tokenizer.json` from the notebook and the server will prefer JSON.
* **Intel macOS and TF install fails**
  Use Python **3.10** and TF **2.10.1** (already pinned). Apple Silicon uses `tensorflow-macos==2.16.1`.

---

## Development tips

* Switch decoding on the fly via `.env` (no code change).
* Want deterministic sampling for demos? You can seed the RNG in `services/predictor.py` where we create `np.random.default_rng()`.
* If you ever change `seq_length` in training, update `metadata.json` so the UI validation and the service stay in sync.


---

## Credits

* Training done in Colab (see `Predict_words.ipynb`).

---

## Acknowledgments

This project benefited from assistance by **ChatGPT (OpenAI, GPT-5 Thinking)**.  
AI support was used to accelerate Django scaffolding, environment pinning for TensorFlow on macOS Intel, 
resilient artifact loading (H5 weights + tokenizer compat), sampling-based decoding (temperature / top-k / top-p with repetition penalty), 
UI tweaks (Clear button), test scaffolding, and this README.

All code was reviewed and integrated by the project author. Any mistakes are ours.

---

### License

[MIT LICENSE](LICENSE)
