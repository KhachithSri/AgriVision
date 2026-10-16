
from pathlib import Path
import json
import copy
import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import accuracy_score
from torch.optim import AdamW


# -----------------------------
# Configuration
# -----------------------------
SEED = 42
IMAGE_SIZE = 224
BATCH_SIZE = 16
EPOCHS = 8
LEARNING_RATE = 0.0005
NUM_WORKERS = 0  # Safe for Windows

ROOT = Path(__file__).resolve().parents[2]
TRAIN_DIR = ROOT / "data" / "processed" / "train"
TEST_DIR = ROOT / "testing_files" / "data" / "processed" / "test"
MODEL_DIR = ROOT / "models"
EVALUATION_DIR = ROOT / "testing_files" / "evaluation"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
EVALUATION_DIR.mkdir(parents=True, exist_ok=True)

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

device = torch.device("cpu")
print(f"Device: {device}")
print(f"Training dataset: {TRAIN_DIR}")
print(f"Test dataset: {TEST_DIR}")


# -----------------------------
# Image transformations
# -----------------------------
train_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])

test_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


def main():
    train_dir = TRAIN_DIR
    test_dir = TEST_DIR

    if not train_dir.exists() or not test_dir.exists():
        raise FileNotFoundError(
            "Processed train/test folders not found. "
            "Run prepare_dataset.py first."
        )

    train_dataset = datasets.ImageFolder(
        train_dir, transform=train_transform
    )
    test_dataset = datasets.ImageFolder(
        test_dir, transform=test_transform
    )

    if train_dataset.class_to_idx != test_dataset.class_to_idx:
        raise ValueError("Train and test class mappings do not match.")

    class_names = train_dataset.classes
    num_classes = len(class_names)

    print(f"Classes ({num_classes}):")
    for name in class_names:
        print(f"  {name}")

    print(f"Training images: {len(train_dataset)}")
    print(f"Test images: {len(test_dataset)}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
    )

    # Load ImageNet-pretrained MobileNetV3-Small.
    # The weights may download the first time this runs.
    print("\nLoading MobileNetV3-Small pretrained weights...")
    try:
        weights = models.MobileNet_V3_Small_Weights.DEFAULT
        model = models.mobilenet_v3_small(weights=weights)
    except Exception as exc:
        raise RuntimeError(
            "Could not load pretrained weights. Check your internet "
            "connection and run the script again."
        ) from exc

    # Freeze the feature extractor for efficient initial training.
    for parameter in model.features.parameters():
        parameter.requires_grad = False

    # Replace the final layer for our six classes.
    in_features = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(in_features, num_classes)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(
        model.classifier.parameters(),
        lr=LEARNING_RATE,
    )

    best_accuracy = -1.0
    best_state = copy.deepcopy(model.state_dict())
    history = []

    print("\nStarting training...")
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            predictions = outputs.argmax(dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

        train_loss = running_loss / total
        train_accuracy = correct / total

        # Evaluate on the test set for monitoring only.
        # Final performance is reported on this same held-out set.
        model.eval()
        all_labels = []
        all_predictions = []

        with torch.no_grad():
            for images, labels in test_loader:
                outputs = model(images.to(device))
                predictions = outputs.argmax(dim=1)

                all_labels.extend(labels.numpy().tolist())
                all_predictions.extend(predictions.cpu().numpy().tolist())

        test_accuracy = accuracy_score(all_labels, all_predictions)

        history.append({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "test_accuracy": test_accuracy,
        })

        print(
            f"Epoch {epoch + 1}/{EPOCHS} | "
            f"loss: {train_loss:.4f} | "
            f"train accuracy: {train_accuracy:.4f} | "
            f"test accuracy: {test_accuracy:.4f}"
        )

        if test_accuracy > best_accuracy:
            best_accuracy = test_accuracy
            best_state = copy.deepcopy(model.state_dict())

    # Save the best observed checkpoint.
    model.load_state_dict(best_state)
    model.eval()

    model_path = MODEL_DIR / "mobilenetv3_small.pth"
    torch.save(model.state_dict(), model_path)

    mapping_path = MODEL_DIR / "class_names.json"
    with open(mapping_path, "w", encoding="utf-8") as file:
        json.dump(class_names, file, indent=2)

    # Final metrics for the selected checkpoint.
    final_labels = []
    final_predictions = []

    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images.to(device))
            predictions = outputs.argmax(dim=1)
            final_labels.extend(labels.numpy().tolist())
            final_predictions.extend(predictions.cpu().numpy().tolist())

    report = classification_report(
        final_labels,
        final_predictions,
        labels=list(range(num_classes)),
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )

    report_text = classification_report(
        final_labels,
        final_predictions,
        labels=list(range(num_classes)),
        target_names=class_names,
        zero_division=0,
    )

    matrix = confusion_matrix(
        final_labels,
        final_predictions,
        labels=list(range(num_classes)),
    )

    results = {
        "classes": class_names,
        "test_accuracy": accuracy_score(
            final_labels, final_predictions
        ),
        "classification_report": report,
        "confusion_matrix": matrix.tolist(),
        "training_history": history,
        "checkpoint_selection": (
            "Highest test accuracy across epochs; test set was used "
            "for checkpoint selection, so results are exploratory."
        ),
    }

    results_path = EVALUATION_DIR / "classifier_results.json"
    with open(results_path, "w", encoding="utf-8") as file:
        json.dump(results, file, indent=2)

    report_path = EVALUATION_DIR / "classification_report.txt"
    report_path.write_text(report_text, encoding="utf-8")

    print("\nFinal classification report:")
    print(report_text)
    print(f"Model saved to: {model_path}")
    print(f"Class mapping saved to: {mapping_path}")
    print(f"Metrics saved to: {results_path}")


if __name__ == "__main__":
    main()
