"""
Generate real evaluation numbers for every saved model: accuracy, per-class
precision/recall/F1, confusion matrix, and one-vs-rest ROC-AUC.

Assumes a test set laid out like:
    test_dir/
        actinic keratosis/*.jpg
        basal cell carcinoma/*.jpg
        ...

Run:
    python evaluate.py --test_dir path/to/test --out_dir results

Outputs (per model, in out_dir):
    <model_name>_classification_report.txt
    <model_name>_confusion_matrix.png
    <model_name>_roc_curves.png
Plus a combined metrics_summary.json comparing all models.
"""

import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.preprocessing import label_binarize
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from model_utils import has_rescaling_layer

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

MODEL_PATHS = {
    "cnn_model1": "_model1_cnn.h5",
    "cnn_model_aug": "model_aug_dropout_cnn.h5",
    "cnn_model_bal": "model_bal.keras",
    "resnet_cnn": "resnet_model_cnn.h5",
    "resnet_new": "new_resnet_model.keras",
}


def load_test_data(test_dir, model, img_size=(180, 180), batch_size=32):
    # Models with a baked-in Rescaling layer expect raw 0-255 input;
    # models without one expect manually normalized 0-1 input.
    rescale = None if has_rescaling_layer(model) else 1.0 / 255
    datagen = ImageDataGenerator(rescale=rescale)
    gen = datagen.flow_from_directory(
        test_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode="categorical",
        classes=CLASS_LABELS,
        shuffle=False,
    )
    return gen


def evaluate_model(name, model_path, test_dir, out_dir):
    print(f"\n=== Evaluating {name} ({model_path}) ===")
    model = load_model(model_path)
    gen = load_test_data(test_dir, model)
    print(f"  preprocessing: {'raw 0-255 (model has internal Rescaling)' if has_rescaling_layer(model) else 'manual /255 normalization'}")

    y_true = gen.classes
    y_pred_probs = model.predict(gen, verbose=1)
    y_pred = np.argmax(y_pred_probs, axis=1)

    report = classification_report(y_true, y_pred, target_names=CLASS_LABELS, digits=3)
    with open(os.path.join(out_dir, f"{name}_classification_report.txt"), "w") as f:
        f.write(report)
    print(report)

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=CLASS_LABELS, yticklabels=CLASS_LABELS)
    plt.title(f"Confusion Matrix — {name}")
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{name}_confusion_matrix.png"), dpi=150)
    plt.close()

    y_true_bin = label_binarize(y_true, classes=list(range(len(CLASS_LABELS))))
    plt.figure(figsize=(9, 7))
    macro_auc_scores = []
    for i, label in enumerate(CLASS_LABELS):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_pred_probs[:, i])
        auc = roc_auc_score(y_true_bin[:, i], y_pred_probs[:, i])
        macro_auc_scores.append(auc)
        plt.plot(fpr, tpr, label=f"{label} (AUC={auc:.2f})")
    plt.plot([0, 1], [0, 1], "k--", linewidth=1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curves (One-vs-Rest) — {name}")
    plt.legend(fontsize=8, loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{name}_roc_curves.png"), dpi=150)
    plt.close()

    accuracy = float(np.mean(y_pred == y_true))
    return {
        "accuracy": accuracy,
        "macro_auc": float(np.mean(macro_auc_scores)),
        "report": report,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_dir", required=True)
    parser.add_argument("--out_dir", default="results")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    summary = {}
    for name, path in MODEL_PATHS.items():
        if not os.path.exists(path):
            print(f"Skipping {name}: {path} not found")
            continue
        summary[name] = evaluate_model(name, path, args.test_dir, args.out_dir)

    with open(os.path.join(args.out_dir, "metrics_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print("\n=== Summary (accuracy / macro-AUC) ===")
    for name, m in summary.items():
        print(f"{name}: acc={m['accuracy']:.3f}  macro_auc={m['macro_auc']:.3f}")


if __name__ == "__main__":
    main()
