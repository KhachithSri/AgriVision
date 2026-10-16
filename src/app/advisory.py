
from src.vision.symptom_analyzer import analyze_leaf_image
from src.rag.retrieve import retrieve


def generate_advisory(
    image,
    predicted_class,
    confidence,
    embedding_model,
    index,
    documents,
):
    """Generate a focused advisory using classifier, vision, and RAG."""

    parts = predicted_class.split("___", maxsplit=1)
    predicted_crop = parts[0].replace("_", " ").strip().lower()
    predicted_disease = (
        parts[1].replace("_", " ").strip().lower()
        if len(parts) > 1
        else ""
    )

    visual_analysis = analyze_leaf_image(image, predicted_class)

    query = (
        f"{predicted_crop} {predicted_disease} "
        "symptoms and management"
    )

    results = retrieve(
        query,
        embedding_model,
        index,
        documents,
        top_k=min(6, len(documents)),
    )

    # Keep only records for the predicted crop and disease.
    matching = []
    for item in results:
        record_crop = str(item.get("crop", "")).strip().lower()
        record_disease = str(item.get("disease", "")).strip().lower()

        crop_matches = record_crop == predicted_crop
        disease_matches = (
            record_disease == predicted_disease
            or record_disease.replace(" ", "_") == predicted_disease
        )

        if crop_matches and disease_matches:
            matching.append({
                "crop": item.get("crop", predicted_crop),
                "disease": item.get("disease", predicted_disease),
                "score": item.get("score", 0.0),
                "text": item.get("text", ""),
            })

    return {
        "predicted_class": predicted_class,
        "classifier_confidence": float(confidence),
        "visual_analysis": visual_analysis,
        "retrieved_knowledge": matching[:1],
        "notice": (
            "AI-assisted assessment only; this is not a confirmed diagnosis. "
            "Classifier confidence is not a guarantee of correctness. "
            "Retrieval similarity is not a disease probability."
        ),
    }
