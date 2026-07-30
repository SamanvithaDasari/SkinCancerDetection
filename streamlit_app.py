"""
Streamlit demo for HuggingFace Spaces.

Deploy:
  1. Create a new Space on huggingface.co -> SDK: Streamlit
  2. Push this file (as app.py), your model files, gradcam.py, and
     requirements.txt to the Space repo (Space repos are git repos,
     use Git LFS for the .h5/.keras weight files if they're large)
  3. The Space builds automatically and gives you a public URL
"""

import numpy as np
import streamlit as st
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image as kimage

from gradcam import make_gradcam_heatmap, overlay_heatmap
from model_utils import preprocess_for_model

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

MODEL_CONFIG = {
    "cnn_model1": {"path": "_model1_cnn.h5", "weight": 0.46},
    "cnn_model_aug": {"path": "model_aug_dropout_cnn.h5", "weight": 0.44},
    "cnn_model_bal": {"path": "model_bal.keras", "weight": 0.36},
    "resnet_cnn": {"path": "resnet_model_cnn.h5", "weight": 0.33},
    "resnet_new": {"path": "new_resnet_model.keras", "weight": 0.14},
}

IMG_SIZE = (180, 180)

st.set_page_config(page_title="Skin Lesion Classifier", page_icon="🩺", layout="centered")


@st.cache_resource
def load_models():
    models = {}
    for name, cfg in MODEL_CONFIG.items():
        try:
            models[name] = load_model(cfg["path"])
        except Exception as e:
            st.warning(f"Could not load {name}: {e}")
    return models


def resize_to_array(pil_img):
    resized = pil_img.convert("RGB").resize(IMG_SIZE)
    arr = kimage.img_to_array(resized)
    return np.expand_dims(arr, axis=0)


st.title("🩺 Skin Lesion Classifier")
st.caption(
    "9-class dermoscopic image classifier using a 5-model CNN/ResNet50 ensemble, "
    "with Grad-CAM explainability."
)
st.warning(
    "⚠️ Research / portfolio prototype only. This is **not** a medical device "
    "and must never be used for real diagnostic decisions.",
    icon="⚠️",
)

models = load_models()
if not models:
    st.error("No models could be loaded. Check that model files are present in the Space repo.")
    st.stop()

uploaded = st.file_uploader("Upload a dermoscopic image", type=["png", "jpg", "jpeg"])

if uploaded is not None:
    pil_img = Image.open(uploaded)
    st.image(pil_img, caption="Uploaded image", width=300)

    with st.spinner("Running ensemble inference..."):
        img_array_raw = resize_to_array(pil_img)

        weighted_probs = np.zeros(len(CLASS_LABELS))
        total_weight = 0.0
        best_model_name, best_weight = None, -1
        rows = []

        for name, model in models.items():
            weight = MODEL_CONFIG[name]["weight"]
            img_array = preprocess_for_model(img_array_raw, model)
            preds = model.predict(img_array, verbose=0)[0]
            weighted_probs += preds * weight
            total_weight += weight
            idx = int(np.argmax(preds))
            rows.append({"model": name, "prediction": CLASS_LABELS[idx], "confidence": f"{preds[idx]:.2%}"})
            if weight > best_weight:
                best_weight, best_model_name = weight, name

        weighted_probs /= total_weight
        ensemble_idx = int(np.argmax(weighted_probs))

    st.subheader("Ensemble prediction")
    st.metric(label="Predicted class", value=CLASS_LABELS[ensemble_idx],
              delta=f"{weighted_probs[ensemble_idx]:.1%} confidence")

    st.subheader("Class probabilities")
    st.bar_chart({CLASS_LABELS[i]: float(weighted_probs[i]) for i in range(len(CLASS_LABELS))})

    st.subheader("Grad-CAM: what the model looked at")
    try:
        gradcam_model = models[best_model_name]
        gradcam_input = preprocess_for_model(img_array_raw, gradcam_model)
        heatmap = make_gradcam_heatmap(gradcam_input, gradcam_model)
        overlaid = overlay_heatmap(pil_img.convert("RGB").resize(IMG_SIZE), heatmap)
        st.image(overlaid, caption=f"Grad-CAM overlay ({best_model_name})", width=300)
    except Exception as e:
        st.info(f"Grad-CAM unavailable for this model: {e}")

    with st.expander("Per-model breakdown"):
        st.table(rows)
