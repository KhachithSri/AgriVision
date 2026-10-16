
import json
from pathlib import Path

import streamlit as st
import torch
from PIL import Image
from torchvision import models, transforms

from src.app.advisory import generate_advisory
from src.rag.retrieve import load_resources


ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = ROOT / "models" / "mobilenetv3_small.pth"
CLASSES_PATH = ROOT / "models" / "class_names.json"

IMAGE_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


@st.cache_resource
def load_classifier():
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

    if not CLASSES_PATH.is_file():
        raise FileNotFoundError(f"Class names not found: {CLASSES_PATH}")

    with open(CLASSES_PATH, "r", encoding="utf-8") as file:
        class_names = json.load(file)

    if isinstance(class_names, dict):
        class_names = [
            class_names[str(i)] for i in range(len(class_names))
        ]

    model = models.mobilenet_v3_small(weights=None)
    model.classifier[3] = torch.nn.Linear(
        model.classifier[3].in_features,
        len(class_names),
    )

    state = torch.load(
        MODEL_PATH,
        map_location="cpu",
        weights_only=True,
    )
    model.load_state_dict(state)
    model.eval()

    return model, class_names


@st.cache_resource
def load_rag():
    return load_resources()


def predict_image(image, model, class_names):
    image_tensor = IMAGE_TRANSFORM(
        image.convert("RGB")
    ).unsqueeze(0)

    with torch.no_grad():
        probabilities = torch.softmax(model(image_tensor), dim=1)[0]

    count = min(3, len(class_names))
    values, indices = torch.topk(probabilities, count)

    predictions = [
        {
            "class": class_names[int(index)],
            "confidence": float(value),
        }
        for value, index in zip(values, indices)
    ]

    best = predictions[0]
    return best["class"], best["confidence"], predictions


def display_class_name(class_name):
    return class_name.replace("___", " — ").replace("_", " ")


def main():
    st.set_page_config(
        page_title="AgriVision AI",
        page_icon="🌿",
        layout="wide",
    )

    st.title("AgriVision AI")
    st.write(
        "Plant health assessment powered by MobileNetV3, Gemma 3 4B, "
        "and a plant disease knowledge base."
    )

    st.info(
        "This tool provides an AI-assisted assessment, not a confirmed "
        "diagnosis. Verify important decisions with a local agricultural expert."
    )

    uploaded_file = st.file_uploader(
        "Upload a plant leaf image",
        type=["jpg", "jpeg", "png"],
        key="leaf_uploader",
    )

    if uploaded_file is None:
        st.caption(
            "Supported classes: Apple scab, Grape black rot, Potato early "
            "blight, Tomato early blight, Tomato late blight, and healthy tomato."
        )
        st.stop()

    try:
        image = Image.open(uploaded_file).convert("RGB")
    except Exception as error:
        st.error(f"Could not open the image: {error}")
        st.stop()

    st.image(image, caption="Uploaded leaf image", width=400)

    with st.spinner("Loading classifier and knowledge base..."):
        try:
            model, class_names = load_classifier()
            embedding_model, index, documents = load_rag()
        except Exception as error:
            st.error(f"Could not load project resources: {error}")
            st.stop()

    with st.spinner("Classifying the image..."):
        try:
            predicted_class, confidence, predictions = predict_image(
                image, model, class_names
            )
        except Exception as error:
            st.error(f"Classification failed: {error}")
            st.stop()

    st.subheader("1. Classifier result")
    st.write(f"**Predicted class:** {display_class_name(predicted_class)}")
    st.write(f"**Model confidence score:** {confidence:.1%}")

    st.caption(
        "This score comes from the model's softmax output. It is not a "
        "calibrated probability of a correct diagnosis."
    )

    # These are provisional display thresholds, not validated confidence levels.
    if confidence >= 0.85:
        st.success(
            "The model strongly favors this class. The prediction still "
            "needs to agree with the image and other evidence."
        )
    elif confidence >= 0.60:
        st.warning(
            "The model's preference is moderate. Compare the visible "
            "symptoms carefully before acting."
        )
    else:
        st.warning(
            "The model's preference is low. Try a clearer image and seek "
            "expert guidance before disease-specific treatment."
        )

    with st.expander("View top three predictions"):
        for rank, item in enumerate(predictions, start=1):
            label = display_class_name(item["class"])
            st.write(
                f"{rank}. **{label}** — {item['confidence']:.2%}"
            )
            st.progress(max(0.0, min(1.0, item["confidence"])))

    st.divider()
    st.subheader("2. AI symptom analysis and advisory")

    # A new image must not display an old image's advisory.
    image_key = (
        f"{uploaded_file.name}_{len(uploaded_file.getvalue())}_"
        f"{predicted_class}"
    )

    if st.session_state.get("advisory_image_key") != image_key:
        st.session_state.pop("latest_advisory", None)
        st.session_state["advisory_image_key"] = image_key

    if st.button(
        "Analyze symptoms and generate advisory",
        type="primary",
        key="generate_advisory_button",
    ):
        with st.spinner(
            "Gemma is examining the leaf and retrieving relevant guidance..."
        ):
            try:
                advisory = generate_advisory(
                    image=image,
                    predicted_class=predicted_class,
                    confidence=confidence,
                    embedding_model=embedding_model,
                    index=index,
                    documents=documents,
                )
                st.session_state["latest_advisory"] = advisory
                st.session_state["advisory_image_key"] = image_key
            except Exception as error:
                st.error(f"Advisory generation failed: {error}")

    advisory = st.session_state.get("latest_advisory")

    if (
        advisory
        and st.session_state.get("advisory_image_key") == image_key
    ):
        st.markdown("#### Visible symptoms and consistency check")
        st.markdown(advisory.get("visual_analysis", "No analysis returned."))

        knowledge = advisory.get("retrieved_knowledge", [])

        st.markdown("#### Relevant agricultural guidance")

        if knowledge:
            for item in knowledge:
                st.markdown(
                    f"**{item.get('crop', 'Crop')} — "
                    f"{item.get('disease', 'Disease')}**"
                )
                st.write(item.get("text", "No guidance text available."))

                score = item.get("score")
                if score is not None:
                    st.caption(
                        f"Knowledge retrieval similarity: {score:.3f}. "
                        "This is not a disease probability."
                    )

            st.markdown("#### Recommended next steps")
            st.write(
                "Compare the observed symptoms with the guidance above. "
                "Remove fallen or clearly infected plant material where "
                "appropriate, avoid unnecessary chemical treatments, and "
                "follow locally recommended crop-management practices."
            )
        else:
            st.warning(
                "No matching disease guidance was found. Do not choose a "
                "disease-specific treatment based on the prediction alone."
            )

        st.markdown("#### Important limitations")
        st.write(advisory.get("notice", "AI results may be incorrect."))
        st.write(
            "If symptoms spread, damage is severe, or the crop and predicted "
            "disease do not appear to match, consult a local agricultural "
            "extension officer."
        )


if __name__ == "__main__":
    main()
