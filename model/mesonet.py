from __future__ import annotations

import random

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models


IMAGE_SIZE = (256, 256)


def set_global_determinism(seed: int = 42) -> None:
    """Set deterministic behavior across Python, NumPy, and TensorFlow."""
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)



def build_mesonet_like(input_shape: tuple[int, int, int] = (256, 256, 3)) -> tf.keras.Model:
    """
    Lightweight MesoNet-inspired CNN for binary image classification.

    Output is a single sigmoid unit:
      - value near 0.0 -> REAL
      - value near 1.0 -> FAKE
    """
    set_global_determinism(42)

    model = models.Sequential(
        [
            layers.Input(shape=input_shape),
            layers.Rescaling(1.0 / 255.0),
            layers.Conv2D(8, (3, 3), padding="same", activation="relu"),
            layers.BatchNormalization(),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Conv2D(8, (5, 5), padding="same", activation="relu"),
            layers.BatchNormalization(),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Conv2D(16, (5, 5), padding="same", activation="relu"),
            layers.BatchNormalization(),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Conv2D(16, (5, 5), padding="same", activation="relu"),
            layers.BatchNormalization(),
            layers.MaxPooling2D(pool_size=(4, 4)),
            layers.Flatten(),
            layers.Dropout(0.5),
            layers.Dense(16, activation="relu"),
            layers.Dense(1, activation="sigmoid"),
        ]
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model
