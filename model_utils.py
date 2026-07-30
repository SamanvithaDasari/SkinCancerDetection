"""
Shared helper: different models in this project were trained differently —
some have a Rescaling(1./255) layer baked into the model graph (so they
expect raw 0-255 pixel input), others have no such layer and were trained
against manually pre-normalized 0-1 input.

Feeding the wrong one silently produces near-random predictions (this bit
us during evaluation — a 9-class model scoring ~10% accuracy is a
preprocessing bug, not a bad model, almost every time).

This module auto-detects which case a given model falls into so every
script (evaluate.py, gradcam.py, app_v2.py, streamlit_app.py) treats each
model consistently.
"""

import numpy as np


def has_rescaling_layer(model) -> bool:
    """True if the model normalizes its own input (Rescaling/Normalization
    layer present anywhere in the graph, including one level of nesting)."""
    for layer in model.layers:
        cls_name = layer.__class__.__name__.lower()
        if "rescaling" in cls_name or "normalization" in cls_name:
            return True
        # check one level of nesting (e.g. a Sequential wrapping a backbone)
        sub_layers = getattr(layer, "layers", None)
        if sub_layers:
            for sub in sub_layers:
                sub_cls = sub.__class__.__name__.lower()
                if "rescaling" in sub_cls or "normalization" in sub_cls:
                    return True
    return False


def preprocess_for_model(img_array: np.ndarray, model) -> np.ndarray:
    """
    img_array: raw pixel array in 0-255 range, shape (1, H, W, 3) or (H, W, 3)
    Returns the array preprocessed correctly for this specific model.
    """
    img_array = np.asarray(img_array, dtype=np.float32)
    if has_rescaling_layer(model):
        return img_array  # model normalizes internally, feed raw 0-255
    return img_array / 255.0  # model expects pre-normalized input
