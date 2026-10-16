
from pathlib import Path
import json

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


ROOT = Path(__file__).resolve().parents[2]
KNOWLEDGE_PATH = ROOT / "knowledge" / "plant_diseases.json"
INDEX_DIR = ROOT / "knowledge" / "faiss_index"

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def record_to_text(record):
    sections = [
        f"Crop: {record['crop']}",
        f"Disease: {record['disease']}",
        f"Status: {record['status']}",
        "Symptoms: " + "; ".join(record["symptoms"]),
        "Favorable conditions: "
        + "; ".join(record["favorable_conditions"]),
        "Management: " + "; ".join(record["management"]),
    ]
    return "\n".join(sections)


def main():
    if not KNOWLEDGE_PATH.is_file():
        raise FileNotFoundError(
            f"Knowledge file not found: {KNOWLEDGE_PATH}"
        )

    with open(KNOWLEDGE_PATH, "r", encoding="utf-8") as file:
        records = json.load(file)

    if not records:
        raise ValueError("The knowledge file contains no records.")

    class_names = [record["class_name"] for record in records]
    if len(class_names) != len(set(class_names)):
        raise ValueError("Duplicate class_name entries found.")

    documents = [
        {
            "class_name": record["class_name"],
            "crop": record["crop"],
            "disease": record["disease"],
            "status": record["status"],
            "text": record_to_text(record),
        }
        for record in records
    ]

    print(f"Loading embedding model: {EMBEDDING_MODEL}")
    model = SentenceTransformer(EMBEDDING_MODEL)

    texts = [document["text"] for document in documents]
    embeddings = model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    )

    embeddings = np.asarray(embeddings, dtype=np.float32)

    # Inner product equals cosine similarity for normalized vectors.
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    INDEX_DIR.mkdir(parents=True, exist_ok=True)

    faiss.write_index(index, str(INDEX_DIR / "diseases.faiss"))

    with open(
        INDEX_DIR / "documents.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(documents, file, indent=2, ensure_ascii=False)

    with open(
        INDEX_DIR / "index_config.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            {
                "embedding_model": EMBEDDING_MODEL,
                "embedding_dimension": int(embeddings.shape[1]),
                "document_count": len(documents),
                "similarity": "cosine",
            },
            file,
            indent=2,
        )

    print(f"\nIndexed documents: {index.ntotal}")
    print(f"Embedding dimension: {embeddings.shape[1]}")
    print(f"Index saved to: {INDEX_DIR}")


if __name__ == "__main__":
    main()
