"""
Grad-CAM explainability for the skin lesion classifiers.

Works with:
  - plain Sequential/Functional CNNs (finds the last Conv2D layer directly)
  - ResNet50-based models where the backbone is a nested sub-model
    (finds the last conv layer inside the nested backbone and builds
    a two-stage gradient model)

Usage:
    from gradcam import make_gradcam_heatmap, overlay_heatmap

    heatmap = make_gradcam_heatmap(img_array, model)
    overlaid_img = overlay_heatmap(original_pil_image, heatmap)
"""

import numpy as np
import tensorflow as tf
from PIL import Image


def _find_last_conv_layer(model):
    """Return the name of the last Conv2D layer, searching one level into
    nested sub-models (e.g. a ResNet50 backbone wrapped in a Sequential)."""
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer.name, model
        if isinstance(layer, tf.keras.Model):
            for sub_layer in reversed(layer.layers):
                if isinstance(sub_layer, tf.keras.layers.Conv2D):
                    return sub_layer.name, layer
    raise ValueError(
        "No Conv2D layer found in model or its nested sub-models. "
        "Pass last_conv_layer_name explicitly."
    )


def make_gradcam_heatmap(img_array, model, last_conv_layer_name=None, pred_index=None):
    """
    img_array: preprocessed input, shape (1, H, W, 3)
    model: a loaded tf.keras model
    Returns a (H, W) heatmap normalized to [0, 1]
    """
    if last_conv_layer_name is None:
        last_conv_layer_name, conv_owner = _find_last_conv_layer(model)
    else:
        conv_owner = model

    grad_model = tf.keras.models.Model(
        inputs=conv_owner.inputs if conv_owner is not model else model.inputs,
        outputs=[conv_owner.get_layer(last_conv_layer_name).output, conv_owner.output]
        if conv_owner is not model
        else [model.get_layer(last_conv_layer_name).output, model.output],
    )

    with tf.GradientTape() as tape:
        conv_output, predictions = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]

    grads = tape.gradient(class_channel, conv_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_output = conv_output[0]
    heatmap = conv_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy()


def overlay_heatmap(original_image: Image.Image, heatmap: np.ndarray, alpha: float = 0.4) -> Image.Image:
    """Overlay a Grad-CAM heatmap on the original PIL image and return a new PIL image."""
    import matplotlib.cm as cm

    heatmap = np.uint8(255 * heatmap)
    jet = cm.get_cmap("jet")
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]
    jet_heatmap = tf.keras.utils.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((original_image.width, original_image.height))
    jet_heatmap = tf.keras.utils.img_to_array(jet_heatmap)

    original_array = tf.keras.utils.img_to_array(original_image.convert("RGB"))
    superimposed = jet_heatmap * alpha + original_array
    superimposed = tf.keras.utils.array_to_img(superimposed)
    return superimposed


if __name__ == "__main__":
    # Quick smoke test — point this at one of your saved models + a sample image
    import sys
    from tensorflow.keras.models import load_model
    from tensorflow.keras.preprocessing import image as kimage

    from model_utils import preprocess_for_model

    if len(sys.argv) != 3:
        print("Usage: python gradcam.py <model_path> <image_path>")
        sys.exit(1)

    model_path, img_path = sys.argv[1], sys.argv[2]
    model = load_model(model_path)

    pil_img = Image.open(img_path).convert("RGB").resize((180, 180))
    arr = kimage.img_to_array(pil_img)
    arr = np.expand_dims(arr, axis=0)
    arr = preprocess_for_model(arr, model)

    heatmap = make_gradcam_heatmap(arr, model)
    result = overlay_heatmap(pil_img, heatmap)
    result.save("gradcam_output.png")
    print("Saved gradcam_output.png")
