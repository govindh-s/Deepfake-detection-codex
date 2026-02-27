from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from model.mesonet import IMAGE_SIZE, build_mesonet_like, set_global_determinism


BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_WEIGHTS_PATH = BASE_DIR / "model" / "weights.h5"

# Caches model after first load to avoid repeated startup cost.
_MODEL = None


def _load_model(weights_path: Path = DEFAULT_WEIGHTS_PATH):
    global _MODEL
    if _MODEL is not None:
        return _MODEL

    set_global_determinism(42)
    model = build_mesonet_like(input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3))

    if weights_path.exists():
        model.load_weights(str(weights_path))

    _MODEL = model
    return _MODEL



def _preprocess_image(image_path: str) -> np.ndarray:
    path = Path(image_path)
    if not path.exists():
        raise ValueError(f"Image not found: {path}")

    try:
        image = Image.open(path).convert("RGB")
    except Exception as exc:
        raise ValueError("Could not read image. Upload a valid image file.") from exc

    image = image.resize(IMAGE_SIZE)
    arr = np.asarray(image, dtype=np.float32)
    return np.expand_dims(arr, axis=0)



def predict(image_path: str, weights_path: str | None = None) -> dict[str, Any]:
    """
    Predict whether an image is REAL or FAKE.

    Returns:
        {
          "label": "REAL" | "FAKE",
          "confidence": float  # rounded to 4 decimals
        }
    """
    model = _load_model(Path(weights_path) if weights_path else DEFAULT_WEIGHTS_PATH)
    image_tensor = _preprocess_image(image_path)

    score = float(model.predict(image_tensor, verbose=0)[0][0])
    label = "FAKE" if score >= 0.5 else "REAL"
    confidence = score if label == "FAKE" else (1.0 - score)

    return {
        "label": label,
        "confidence": round(float(confidence), 4),
    }
