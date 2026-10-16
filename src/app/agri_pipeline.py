
from pathlib import Path
import json
import sys

import faiss
import numpy as np
import torch
from PIL import Image
from sentence_transformers import SentenceTransformer
from torchvision import models, transforms


ROOT = Path(__file__).resolve().parents[2]

MODEL_PATH = ROOT / "models" / "mobilenetv3_small.pth"
CLASSES_PATH = ROOT / "models" / "class_names.json"
INDEX_DIR = ROOT / "knowledge" / "faiss_index"

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEVICE = torch.device("cpu")

IMAGE_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


def load_classifier():
    with open(CLASSES_PATH, "r", encoding="utf-8") as file:
        class_names = json.load(file)

    model = models.mobilenet_v3_small(weights=None)
    model.classifier[3] = torch.nn.Linear(
        model.classifier[3].in_features,
        len(class_names),
    )

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE,
        weights_only=True,
    )

    # Support either a plain state dict or a checkpoint dictionary.
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        checkpoint = checkpoint["model_state_dict"]

    model.load_state_dict(checkpoint)
    model.to(DEVICE)
    model.eval()

    return model, class_names


def load_knowledge():
    index_path = INDEX_DIR / "diseases.faiss"
    documents_path = INDEX_DIR / "documents.json"

    index = faiss.read_index(str(index_path))

    with open(documents_path, "r", encoding="utf-8") as file:
        documents = json.load(file)

    embedding_model = SentenceTransformer(EMBEDDING_MODEL)
    return index, documents, embedding_model


def predict_image(image_path, model, class_names):
    image_path = Path(image_path)

    if not image_path.is_file():
        raise FileNotFoundError(f"Image not found: {image_path}")

    with Image.open(image_path) as image:
        image = image.convert("RGB")
        tensor = IMAGE_TRANSFORM(image).unsqueeze(0).to(DEVICE)

    with torch.inference_mode():
        probabilities = torch.softmax(model(tensor), dim=1)[0]
        confidence, predicted_index = torch.max(probabilities, dim=0)

    return (
        class_names[predicted_index.item()],
        float(confidence.item()),
    )


def retrieve_disease(predicted_class, index, documents, embedding_model):
    # Prefer an exact classifier-to-knowledge match.
    for document in documents:
        if document["class_name"] == predicted_class:
            query = document["text"]
            break
    else:
        raise ValueError(
            f"No knowledge record found for class: {predicted_class}"
        )

    query_vector = embedding_model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    query_vector = np.asarray(query_vector, dtype=np.float32)

    scores, indices = index.search(query_vector, len(documents))

    for score, idx in zip(scores[0], indices[0]):
        if idx >= 0 and documents[int(idx)]["class_name"] == predicted_class:
            return documents[int(idx)]

    raise ValueError(f"Could not retrieve knowledge for {predicted_class}")


def main():
    if not MODEL_PATH.is_file() or not CLASSES_PATH.is_file():
        raise FileNotFoundError("Classifier model or class names file is missing.")

    model, class_names = load_classifier()
    index, documents, embedding_model = load_knowledge()

    if len(documents) != index.ntotal:
        raise ValueError("FAISS index and knowledge records do not match.")

    image_path = input("Enter the full path to a plant-leaf image: ").strip()
    image_path = image_path.strip('"')

    predicted_class, confidence = predict_image(
        image_path, model, class_names
    )
    knowledge = retrieve_disease(
        predicted_class, index, documents, embedding_model
    )

    print("\n" + "=" * 60)
    print("AGRIVISION AI - PLANT DISEASE ANALYSIS")
    print("=" * 60)
    print(f"Predicted class : {predicted_class}")
    print(f"Model confidence: {confidence:.2%}")
    print(f"Crop            : {knowledge['crop']}")
    print(f"Diagnosis label : {knowledge['disease']}")

    print("\nSymptoms:")
    for item in knowledge["text"].split("Symptoms: ", 1)[1].split(
        "\nFavorable conditions:", 1
    )[0].split("; "):
        print(f"- {item}")

    print("\nDisease information:")
    print(knowledge["text"])

    print("\nDisclaimer: This is an AI prediction, not a confirmed diagnosis.")
    print("Verify results with a local agricultural expert when needed.")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"\nError: {error}", file=sys.stderr)
        sys.exit(1)
