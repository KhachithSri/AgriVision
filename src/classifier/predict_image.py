
from pathlib import Path
import argparse
import json

import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms


ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = ROOT / "models" / "mobilenetv3_small.pth"
CLASS_NAMES_PATH = ROOT / "models" / "class_names.json"

IMAGE_SIZE = 224
DEVICE = torch.device("cpu")

IMAGE_TRANSFORM = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


def main():
    parser = argparse.ArgumentParser(
        description="Predict a plant disease from an image."
    )
    parser.add_argument(
        "image_path",
        help="Path to the plant leaf image.",
    )
    args = parser.parse_args()

    image_path = Path(args.image_path)

    if not image_path.is_file():
        raise FileNotFoundError(
            f"Image not found: {image_path}"
        )

    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"Trained model not found: {MODEL_PATH}"
        )

    if not CLASS_NAMES_PATH.is_file():
        raise FileNotFoundError(
            f"Class names not found: {CLASS_NAMES_PATH}"
        )

    with open(CLASS_NAMES_PATH, "r", encoding="utf-8") as f:
        class_names = json.load(f)

    model = models.mobilenet_v3_small(weights=None)

    in_features = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(
        in_features, len(class_names)
    )

    state_dict = torch.load(
        MODEL_PATH,
        map_location=DEVICE,
        weights_only=True,
    )
    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()

    with Image.open(image_path) as image:
        image = image.convert("RGB")
        image_tensor = IMAGE_TRANSFORM(image).unsqueeze(0)

    with torch.no_grad():
        logits = model(image_tensor.to(DEVICE))
        probabilities = torch.softmax(logits, dim=1)[0]

    top_count = min(3, len(class_names))
    top_probabilities, top_indices = torch.topk(
        probabilities, top_count
    )

    print("\nAgriVision-AI prediction")
    print("-" * 35)

    for probability, index in zip(
        top_probabilities.tolist(),
        top_indices.tolist(),
    ):
        print(
            f"{class_names[index]}: "
            f"{probability * 100:.2f}%"
        )


if __name__ == "__main__":
    main()
