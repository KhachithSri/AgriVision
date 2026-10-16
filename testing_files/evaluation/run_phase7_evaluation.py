"""Reproducible Phase 7 classifier evaluation for the saved AgriVision model.

The default dataset is the held-out PlantVillage split.  An optional external
ImageFolder directory can be supplied only when its class directory names
match models/class_names.json.  External results are saved separately so they
are never confused with the official held-out result.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms


ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = ROOT / "models" / "mobilenetv3_small.pth"
CLASS_NAMES_PATH = ROOT / "models" / "class_names.json"
DEFAULT_TEST_DIR = (
    ROOT / "testing_files" / "data" / "processed" / "test"
)
EVALUATION_DIR = ROOT / "testing_files" / "evaluation"
DEVICE = torch.device("cpu")

TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


def load_model(class_names: list[str]) -> torch.nn.Module:
    model = models.mobilenet_v3_small(weights=None)
    model.classifier[3] = nn.Linear(
        model.classifier[3].in_features, len(class_names)
    )
    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        checkpoint = checkpoint["model_state_dict"]
    model.load_state_dict(checkpoint)
    return model.to(DEVICE).eval()


def evaluate(dataset_dir: Path, output_stem: str) -> dict:
    if not dataset_dir.is_dir():
        raise FileNotFoundError(f"Dataset folder not found: {dataset_dir}")
    if not MODEL_PATH.is_file() or not CLASS_NAMES_PATH.is_file():
        raise FileNotFoundError("Saved model or class-name file is missing.")

    class_names = json.loads(CLASS_NAMES_PATH.read_text(encoding="utf-8"))
    dataset = datasets.ImageFolder(dataset_dir, transform=TRANSFORM)
    if dataset.classes != class_names:
        raise ValueError(
            "Class folders must exactly match models/class_names.json. "
            f"Expected {class_names}; found {dataset.classes}."
        )

    loader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=0)
    model = load_model(class_names)
    true_labels: list[int] = []
    predictions: list[int] = []
    confidence_scores: list[float] = []

    with torch.inference_mode():
        for images, labels in loader:
            probabilities = torch.softmax(model(images.to(DEVICE)), dim=1)
            confidence, predicted = probabilities.max(dim=1)
            true_labels.extend(labels.tolist())
            predictions.extend(predicted.cpu().tolist())
            confidence_scores.extend(confidence.cpu().tolist())

    label_ids = list(range(len(class_names)))
    report = classification_report(
        true_labels, predictions, labels=label_ids, target_names=class_names,
        output_dict=True, zero_division=0,
    )
    matrix = confusion_matrix(true_labels, predictions, labels=label_ids)
    correct_predictions = [
        predicted == actual
        for actual, predicted in zip(true_labels, predictions)
    ]
    confidence_bands = {}
    for band, lower, upper in (
        ("low", 0.0, 0.60),
        ("moderate", 0.60, 0.85),
        ("high", 0.85, 1.01),
    ):
        selected = [
            lower <= score < upper
            for score in confidence_scores
        ]
        selected_count = sum(selected)
        confidence_bands[band] = {
            "lower_bound_inclusive": lower,
            "upper_bound_exclusive": upper,
            "image_count": selected_count,
            "accuracy": (
                sum(
                    correct
                    for correct, is_selected in zip(
                        correct_predictions, selected
                    )
                    if is_selected
                ) / selected_count
                if selected_count
                else None
            ),
        }

    results = {
        "evaluation_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_directory": str(dataset_dir.resolve()),
        "dataset_type": "external" if output_stem == "external" else "held_out_test",
        "classes": class_names,
        "image_count": len(dataset),
        "accuracy": accuracy_score(true_labels, predictions),
        "macro_precision": report["macro avg"]["precision"],
        "macro_recall": report["macro avg"]["recall"],
        "macro_f1": report["macro avg"]["f1-score"],
        "confidence_score_note": (
            "Maximum softmax scores are uncalibrated classifier scores, "
            "not probabilities of correctness."
        ),
        "confidence_bands": confidence_bands,
        "per_class": {
            name: report[name] for name in class_names
        },
        "confusion_matrix": matrix.tolist(),
    }

    EVALUATION_DIR.mkdir(exist_ok=True)
    json_path = EVALUATION_DIR / f"{output_stem}_evaluation.json"
    json_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    figure, axis = plt.subplots(figsize=(9, 7))
    image = axis.imshow(matrix, cmap="Blues")
    figure.colorbar(image, ax=axis, label="Image count")
    axis.set(
        xticks=np.arange(len(class_names)), yticks=np.arange(len(class_names)),
        xticklabels=class_names, yticklabels=class_names,
        xlabel="Predicted class", ylabel="True class",
        title=f"{results['dataset_type'].replace('_', ' ').title()} confusion matrix",
    )
    plt.setp(axis.get_xticklabels(), rotation=35, ha="right")
    for row, column in np.ndindex(matrix.shape):
        axis.text(column, row, str(matrix[row, column]), ha="center", va="center")
    figure.tight_layout()
    figure.savefig(EVALUATION_DIR / f"{output_stem}_confusion_matrix.png", dpi=180)
    plt.close(figure)

    print(json.dumps({
        key: results[key] for key in (
            "dataset_type", "image_count", "accuracy", "macro_precision",
            "macro_recall", "macro_f1",
        )
    }, indent=2))
    print(f"Saved: {json_path}")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Run AgriVision Phase 7 evaluation.")
    parser.add_argument(
        "--external-dir", type=Path,
        help="Optional labelled external ImageFolder; results are kept separate.",
    )
    args = parser.parse_args()
    evaluate(DEFAULT_TEST_DIR, "held_out_test")
    if args.external_dir:
        evaluate(args.external_dir, "external")


if __name__ == "__main__":
    main()
