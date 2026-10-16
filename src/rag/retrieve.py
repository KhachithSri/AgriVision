
from pathlib import Path
import json

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


ROOT = Path(__file__).resolve().parents[2]
INDEX_DIR = ROOT / "knowledge" / "faiss_index"

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def load_resources():
    index_path = INDEX_DIR / "diseases.faiss"
    documents_path = INDEX_DIR / "documents.json"

    if not index_path.is_file() or not documents_path.is_file():
        raise FileNotFoundError(
            "FAISS index files not found. Run src/rag/build_index.py first."
        )

    index = faiss.read_index(str(index_path))

    with open(documents_path, "r", encoding="utf-8") as file:
        documents = json.load(file)

    model = SentenceTransformer(EMBEDDING_MODEL)
    return model, index, documents


def retrieve(query, model, index, documents, top_k=3):
    query_embedding = model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    query_embedding = np.asarray(query_embedding, dtype=np.float32)

    top_k = min(top_k, len(documents))
    scores, indices = index.search(query_embedding, top_k)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx < 0:
            continue

        results.append({
            "score": float(score),
            **documents[int(idx)],
        })

    return results


def main():
    model, index, documents = load_resources()

    query = input(
        "Enter a plant disease question (e.g., "
        "'What are the symptoms of tomato early blight?'): "
    ).strip()

    if not query:
        print("Please enter a question.")
        return

    results = retrieve(query, model, index, documents)

    print("\nRetrieved knowledge")
    print("-" * 60)

    for number, result in enumerate(results, start=1):
        print(f"\n{number}. {result['crop']} - {result['disease']}")
        print(f"Similarity score: {result['score']:.4f}")
        print(result["text"])


if __name__ == "__main__":
    main()
