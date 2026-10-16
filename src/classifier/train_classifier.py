
from pathlib import Path
import copy
import json
import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms, models
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import accuracy_score
from torch.optim import AdamW


# Configuration
SEED = 42
IMAGE_SIZE = 224
BATCH_SIZE = 16
EPOCHS = 8
LEARNING_RATE = 0.0005
NUM_WORKERS = 0

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


def main():
    train_dir = TRAIN_DIR
    test_dir = TEST_DIR

    if not train_dir.exists() or not test_dir.exists():
        raise FileNotFoundError("Processed dataset folders are missing.")

    # Separate datasets so validation images do not receive augmentation.
    augmented_dataset = datasets.ImageFolder(
        train_dir,
        transform=transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]),
    )

    plain_training_dataset = datasets.ImageFolder(
        train_dir,
        transform=transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]),
    )

    test_dataset = datasets.ImageFolder(
        test_dir,
        transform=transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]),
    )

    class_names = augmented_dataset.classes
    num_classes = len(class_names)

    if plain_training_dataset.class_to_idx != test_dataset.class_to_idx:
        raise ValueError("Train/test class mappings do not match.")

    labels = np.array(augmented_dataset.targets)
    indices = np.arange(len(labels))

    # Stratified split: maintain class proportions in each subset.
    train_indices, val_indices = train_test_split(
        indices,
        test_size=0.20,
        random_state=SEED,
        stratify=labels,
    )

    train_dataset = Subset(augmented_dataset, train_indices)
    val_dataset = Subset(plain_training_dataset, val_indices)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
    )

    print(f"Device: {device}")
    print(f"Classes: {class_names}")
    print(f"Training images: {len(train_dataset)}")
    print(f"Validation images: {len(val_dataset)}")
    print(f"Held-out test images: {len(test_dataset)}")

    print("\nLoading pretrained MobileNetV3-Small...")
    weights = models.MobileNet_V3_Small_Weights.DEFAULT
    model = models.mobilenet_v3_small(weights=weights)

    # Freeze the feature extractor; train the new classification head.
    for parameter in model.features.parameters():
        parameter.requires_grad = False

    in_features = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(in_features, num_classes)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(
        model.classifier.parameters(),
        lr=LEARNING_RATE,
    )

    best_val_accuracy = -1.0
    best_state = copy.deepcopy(model.state_dict())
    history = []

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels_batch in train_loader:
            images = images.to(device)
            labels_batch = labels_batch.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels_batch)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            correct += (
                outputs.argmax(dim=1) == labels_batch
            ).sum().item()
            total += labels_batch.size(0)

        train_loss = running_loss / total
        train_accuracy = correct / total

        # Select checkpoints using validation data only.
        model.eval()
        val_labels = []
        val_predictions = []

        with torch.no_grad():
            for images, labels_batch in val_loader:
                outputs = model(images.to(device))
                predictions = outputs.argmax(dim=1)
                val_labels.extend(labels_batch.numpy().tolist())
                val_predictions.extend(predictions.cpu().numpy().tolist())

        val_accuracy = accuracy_score(val_labels, val_predictions)

        history.append({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "validation_accuracy": val_accuracy,
        })

        print(
            f"Epoch {epoch + 1}/{EPOCHS} | "
            f"loss: {train_loss:.4f} | "
            f"train accuracy: {train_accuracy:.4f} | "
            f"validation accuracy: {val_accuracy:.4f}"
        )

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_state = copy.deepcopy(model.state_dict())

    # Restore the best validation checkpoint.
    model.load_state_dict(best_state)
    model.eval()

    # Save model and matching class names.
    model_path = MODEL_DIR / "mobilenetv3_small.pth"
    torch.save(model.state_dict(), model_path)

    with open(MODEL_DIR / "class_names.json", "w", encoding="utf-8") as f:
        json.dump(class_names, f, indent=2)

    # Evaluate the official test set once, after model selection.
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
    )

    true_labels = []
    predictions = []

    with torch.no_grad():
        for images, labels_batch in test_loader:
            outputs = model(images.to(device))
            predicted = outputs.argmax(dim=1)
            true_labels.extend(labels_batch.numpy().tolist())
            predictions.extend(predicted.cpu().numpy().tolist())

    test_accuracy = accuracy_score(true_labels, predictions)

    report_text = classification_report(
        true_labels,
        predictions,
        labels=list(range(num_classes)),
        target_names=class_names,
        zero_division=0,
    )

    report_dict = classification_report(
        true_labels,
        predictions,
        labels=list(range(num_classes)),
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )

    matrix = confusion_matrix(
        true_labels,
        predictions,
        labels=list(range(num_classes)),
    )

    results = {
        "classes": class_names,
        "training_images": len(train_dataset),
        "validation_images": len(val_dataset),
        "test_images": len(test_dataset),
        "best_validation_accuracy": best_val_accuracy,
        "final_test_accuracy": test_accuracy,
        "classification_report": report_dict,
        "confusion_matrix": matrix.tolist(),
        "training_history": history,
        "checkpoint_selection": "Validation accuracy",
    }

    with open(
        EVALUATION_DIR / "classifier_results.json",
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(results, f, indent=2)

    (EVALUATION_DIR / "classification_report.txt").write_text(
        report_text,
        encoding="utf-8",
    )

    print("\nFinal held-out test classification report:")
    print(report_text)
    print(f"Best validation accuracy: {best_val_accuracy:.4f}")
    print(f"Final test accuracy: {test_accuracy:.4f}")
    print(f"Model saved to: {model_path}")
    print(f"Metrics saved to: {EVALUATION_DIR / 'classifier_results.json'}")


if __name__ == "__main__":
    main()
