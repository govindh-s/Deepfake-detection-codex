# Deepfake Image Detection Web App (Flask + TensorFlow)

A clean, lightweight final-year project for **image-only deepfake detection**.

This project uses:
- **Flask** backend
- **MesoNet-inspired CNN** in TensorFlow/Keras (CPU)
- Image upload via browser
- Prediction output: `REAL` / `FAKE` + confidence score

> Note: This is intentionally lightweight and deployment-focused. It demonstrates architecture and inference flow, not research-grade accuracy.

---

## Project Structure

```text
Deepfake-detection-codex/
├── app.py
├── model/
│   ├── mesonet.py
│   ├── predict.py
│   └── weights.h5        # optional (not required to run)
├── static/
│   └── uploads/
├── templates/
│   └── index.html
├── requirements.txt
├── .env.example
└── README.md
```

---

## How It Works

1. User uploads an image from `index.html`.
2. Flask endpoint `/predict` validates file type and size.
3. File is saved in `static/uploads/`.
4. `predict()` preprocesses image to `(256, 256, 3)`.
5. Model is loaded once (cached in memory):
   - If `model/weights.h5` exists, weights are loaded.
   - If not, model still runs with deterministic seeded initialization.
6. Output is returned as JSON:

```json
{
  "label": "REAL",
  "confidence": 0.8123
}
```

---

## Deterministic Behavior (No Random Coin Flip)

When pretrained weights are unavailable, this project still gives **consistent outputs** for the same input by setting fixed seeds for:
- Python random
- NumPy
- TensorFlow

That means predictions are reproducible between runs (same environment + same versions).

---

## Run Locally

### 1) Create virtual environment

```bash
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# .venv\Scripts\activate   # Windows
```

### 2) Install dependencies

```bash
pip install -r requirements.txt
```

### 3) Start app

```bash
python app.py
```

Open: `http://localhost:5000`

---

## API Endpoint

### `POST /predict`

- **Content-Type:** `multipart/form-data`
- **Field name:** `image`

Example using curl:

```bash
curl -X POST -F "image=@sample.jpg" http://localhost:5000/predict
```

---

## Deployment Notes

- Works on CPU-only environments (no GPU required).
- Suitable for Render, Railway, or VM deployment.
- Keep `FLASK_DEBUG=false` for production.
- Add real trained `weights.h5` later for better accuracy without changing API.

---

## Why This Design Is Stable

- Small codebase and clear module separation (`app.py`, `mesonet.py`, `predict.py`).
- Defensive error handling for invalid files and prediction failures.
- Lazy model loading + in-memory cache avoids repeated model rebuild.
- Deterministic seeds avoid unpredictable behavior when weights are missing.

