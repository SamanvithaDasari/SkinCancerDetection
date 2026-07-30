"""
Revamped Flask backend for the skin lesion classifier.

Improvements over the original app.py:
  - Weighted soft-voting ensemble (one final answer, not 5 disconnected ones)
  - Grad-CAM heatmap returned alongside each prediction
  - Input validation (file type, size) and structured error responses
  - Logging instead of print()
  - debug mode controlled by an environment variable, off by default
  - Model paths and ensemble weights externalized to config

Set per-model weights based on validation accuracy from evaluate.py —
better models should count for more in the ensemble.
"""

import base64
import io
import logging
import os

import numpy as np
from flask import Flask, jsonify, render_template, request
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image as kimage

from gradcam import make_gradcam_heatmap, overlay_heatmap
from model_utils import preprocess_for_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}
MAX_FILE_SIZE_MB = 8
IMG_SIZE = (180, 180)

CLASS_LABELS = [
    "actinic keratosis",
    "basal cell carcinoma",
    "dermatofibroma",
    "melanoma",
    "nevus",
    "pigmented benign keratosis",
    "seborrheic keratosis",
    "squamous cell carcinoma",
    "vascular lesion",
]

# Fill these in with real numbers from evaluate.py's metrics_summary.json.
# Equal weights are a safe default until you have measured accuracy.
MODEL_CONFIG = {
    "cnn_model1": {"path": "_model1_cnn.h5", "weight": 0.46},
    "cnn_model_aug": {"path": "model_aug_dropout_cnn.h5", "weight": 0.44},
    "cnn_model_bal": {"path": "model_bal.keras", "weight": 0.36},
    "resnet_cnn": {"path": "resnet_model_cnn.h5", "weight": 0.33},
    "resnet_new": {"path": "new_resnet_model.keras", "weight": 0.14},
}

app = Flask(__name__, template_folder="templates")

logger.info("Loading models...")
MODELS = {}
for name, cfg in MODEL_CONFIG.items():
    if os.path.exists(cfg["path"]):
        MODELS[name] = load_model(cfg["path"])
        logger.info("Loaded %s from %s", name, cfg["path"])
    else:
        logger.warning("Model file not found, skipping: %s", cfg["path"])


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def resize_to_array(pil_img: Image.Image) -> np.ndarray:
    """Resize + convert to array only. Per-model normalization (raw 0-255 vs
    /255) is applied separately for each model since they were trained
    differently — see model_utils.preprocess_for_model."""
    resized = pil_img.convert("RGB").resize(IMG_SIZE)
    arr = kimage.img_to_array(resized)
    return np.expand_dims(arr, axis=0)


def image_to_base64(pil_img: Image.Image) -> str:
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


@app.route("/")
def index():
    return render_template("indexx.html")


@app.route("/health")
def health():
    return jsonify({"status": "ok", "models_loaded": list(MODELS.keys())})


@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    if not allowed_file(file.filename):
        return jsonify({"error": f"Unsupported file type. Allowed: {ALLOWED_EXTENSIONS}"}), 400

    file.seek(0, os.SEEK_END)
    size_mb = file.tell() / (1024 * 1024)
    file.seek(0)
    if size_mb > MAX_FILE_SIZE_MB:
        return jsonify({"error": f"File too large ({size_mb:.1f} MB). Max {MAX_FILE_SIZE_MB} MB"}), 400

    try:
        pil_img = Image.open(file.stream)
        pil_img.load()
    except Exception as exc:
        logger.exception("Failed to read image")
        return jsonify({"error": f"Could not read image: {exc}"}), 400

    img_array_raw = resize_to_array(pil_img)

    per_model_results = []
    weighted_probs = np.zeros(len(CLASS_LABELS))
    total_weight = 0.0
    best_model_name, best_weight = None, -1

    for name, model in MODELS.items():
        weight = MODEL_CONFIG[name]["weight"]
        img_array = preprocess_for_model(img_array_raw, model)
        preds = model.predict(img_array, verbose=0)[0]
        weighted_probs += preds * weight
        total_weight += weight

        idx = int(np.argmax(preds))
        per_model_results.append({
            "model": name,
            "predicted_class": CLASS_LABELS[idx],
            "probability": float(preds[idx]),
            "weight": weight,
        })
        if weight > best_weight:
            best_weight, best_model_name = weight, name

    if total_weight > 0:
        weighted_probs /= total_weight

    ensemble_idx = int(np.argmax(weighted_probs))
    ensemble_result = {
        "predicted_class": CLASS_LABELS[ensemble_idx],
        "confidence": float(weighted_probs[ensemble_idx]),
        "class_probabilities": {CLASS_LABELS[i]: float(weighted_probs[i]) for i in range(len(CLASS_LABELS))},
    }

    gradcam_b64 = None
    try:
        gradcam_model = MODELS.get(best_model_name)
        if gradcam_model is not None:
            gradcam_input = preprocess_for_model(img_array_raw, gradcam_model)
            heatmap = make_gradcam_heatmap(gradcam_input, gradcam_model)
            overlaid = overlay_heatmap(pil_img.convert("RGB").resize(IMG_SIZE), heatmap)
            gradcam_b64 = image_to_base64(overlaid)
    except Exception:
        logger.exception("Grad-CAM generation failed, continuing without it")

    return jsonify({
        "ensemble_prediction": ensemble_result,
        "model_predictions": per_model_results,
        "gradcam_image_base64": gradcam_b64,
        "disclaimer": (
            "This tool is a research/portfolio prototype and is NOT a medical "
            "device. It must not be used for real clinical diagnosis."
        ),
    })


if __name__ == "__main__":
    debug_mode = os.environ.get("FLASK_DEBUG", "false").lower() == "true"
    app.run(debug=debug_mode, host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
