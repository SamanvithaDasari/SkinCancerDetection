"""
Fine-tune a pretrained Vision Transformer (google/vit-base-patch16-224-in21k)
on the same 9-class skin lesion dataset, using HuggingFace transformers.

With ~2000 images total, training a ViT from scratch won't work — this uses
transfer learning (freeze most of the backbone, fine-tune the head + last
few blocks) so a handful of epochs on a Colab GPU is enough for a
respectable result and a legitimate "I used a foundation model" bullet point.

Expects the same folder layout as evaluate.py:
    data_dir/
        train/<class_name>/*.jpg
        test/<class_name>/*.jpg

Run (in Colab, with GPU runtime):
    !pip install transformers datasets accelerate torch torchvision -q
    python train_vit.py --data_dir /content/data --out_dir vit_skin_model --epochs 5
"""

import argparse

import evaluate as hf_evaluate
import numpy as np
import torch
from datasets import load_dataset
from transformers import (
    Trainer,
    TrainingArguments,
    ViTForImageClassification,
    ViTImageProcessor,
)

MODEL_CHECKPOINT = "google/vit-base-patch16-224-in21k"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True, help="Folder with train/ and test/ subfolders")
    parser.add_argument("--out_dir", default="vit_skin_model")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=16)
    args = parser.parse_args()

    dataset = load_dataset("imagefolder", data_dir=args.data_dir)
    labels = dataset["train"].features["label"].names
    id2label = {i: l for i, l in enumerate(labels)}
    label2id = {l: i for i, l in enumerate(labels)}

    processor = ViTImageProcessor.from_pretrained(MODEL_CHECKPOINT)

    def transform(batch):
        images = [img.convert("RGB") for img in batch["image"]]
        inputs = processor(images=images, return_tensors="pt")
        inputs["labels"] = batch["label"]
        return inputs

    dataset = dataset.with_transform(transform)

    model = ViTForImageClassification.from_pretrained(
        MODEL_CHECKPOINT,
        num_labels=len(labels),
        id2label=id2label,
        label2id=label2id,
    )

    # Freeze the backbone, fine-tune only the classifier head + last 2 encoder layers.
    # This is what makes a handful of epochs on ~2000 images actually work.
    for name, param in model.named_parameters():
        if "classifier" in name or "encoder.layer.10" in name or "encoder.layer.11" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False

    accuracy_metric = hf_evaluate.load("accuracy")
    f1_metric = hf_evaluate.load("f1")

    def compute_metrics(eval_pred):
        preds = np.argmax(eval_pred.predictions, axis=1)
        return {
            "accuracy": accuracy_metric.compute(predictions=preds, references=eval_pred.label_ids)["accuracy"],
            "f1_macro": f1_metric.compute(predictions=preds, references=eval_pred.label_ids, average="macro")["f1"],
        }

    def collate_fn(batch):
        return {
            "pixel_values": torch.stack([torch.tensor(x["pixel_values"]) for x in batch]),
            "labels": torch.tensor([x["labels"] for x in batch]),
        }

    training_args = TrainingArguments(
        output_dir=args.out_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=3e-5,
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        logging_steps=10,
        fp16=torch.cuda.is_available(),
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        data_collator=collate_fn,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    metrics = trainer.evaluate()
    print("Final eval metrics:", metrics)

    trainer.save_model(args.out_dir)
    processor.save_pretrained(args.out_dir)
    print(f"Saved fine-tuned ViT to {args.out_dir}")


if __name__ == "__main__":
    main()
