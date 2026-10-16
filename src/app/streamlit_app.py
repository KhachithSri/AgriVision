
import hashlib
import json
from datetime import datetime
from pathlib import Path

import streamlit as st
import torch
from PIL import Image
from torchvision import models, transforms

from src.app.advisory import generate_advisory
from src.app.prediction_policy import (
    HIGH_CONFIDENCE_THRESHOLD,
    INCONCLUSIVE_CONFIDENCE_THRESHOLD,
    is_inconclusive,
)
from src.rag.retrieve import load_resources, retrieve
from src.translation.gemini_translator import (
    GeminiTranslationError,
    translate_text,
)
from src.translation.quality import translation_quality_check


# --------------------------------------------------
# Project configuration
# --------------------------------------------------

ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = ROOT / "models" / "mobilenetv3_small.pth"
CLASS_NAMES_PATH = ROOT / "models" / "class_names.json"

SUPPORTED_CROPS = [
    "Apple",
    "Grape",
    "Potato",
    "Tomato",
    "Unknown / not sure",
]

LANGUAGES = ["English", "Kannada", "Hindi"]


st.set_page_config(
    page_title="AgriVision AI",
    page_icon="🌱",
    layout="wide",
)


# --------------------------------------------------
# Load the trained classifier
# --------------------------------------------------

@st.cache_resource
def load_classifier():
    """Load the trained MobileNetV3-Small classifier."""

    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

    if not CLASS_NAMES_PATH.is_file():
        raise FileNotFoundError(
            f"Class names not found: {CLASS_NAMES_PATH}"
        )

    with open(CLASS_NAMES_PATH, "r", encoding="utf-8") as file:
        class_names = json.load(file)

    model = models.mobilenet_v3_small(weights=None)

    model.classifier[3] = torch.nn.Linear(
        model.classifier[3].in_features,
        len(class_names),
    )

    state_dict = torch.load(
        MODEL_PATH,
        map_location="cpu",
        weights_only=True,
    )

    model.load_state_dict(state_dict)
    model.eval()

    return model, class_names


# --------------------------------------------------
# Load the retrieval resources
# --------------------------------------------------

@st.cache_resource
def load_rag():
    """Load the sentence embedding model and FAISS index."""
    return load_resources()


# --------------------------------------------------
# Image classification
# --------------------------------------------------

def predict_image(image, model, class_names):
    """Predict the top three classes for an uploaded image."""

    preprocess = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )

    tensor = preprocess(image.convert("RGB")).unsqueeze(0)

    with torch.no_grad():
        logits = model(tensor)
        probabilities = torch.softmax(logits, dim=1)[0]

    top_count = min(3, len(class_names))
    values, indices = torch.topk(probabilities, top_count)

    return [
        {
            "class_name": class_names[index],
            "confidence": float(value),
        }
        for value, index in zip(
            values.tolist(),
            indices.tolist(),
        )
    ]


def parse_class_name(class_name):
    """Extract crop and disease labels from a classifier label."""

    parts = class_name.split("___", maxsplit=1)

    crop = parts[0].replace("_", " ").strip()

    disease = (
        parts[1].replace("_", " ").strip()
        if len(parts) > 1
        else "Unknown"
    )

    return crop, disease


def show_prediction_history(container):
    """Render session-only prediction summaries without storing images."""

    with container.container():
        st.subheader("Prediction history")
        st.caption(
            "Stored only for this Streamlit session; uploaded images are not saved here."
        )

        if not st.session_state["agri_prediction_history"]:
            st.caption("No predictions in this session yet.")
            return

        for entry in reversed(st.session_state["agri_prediction_history"]):
            status = "Inconclusive" if entry["inconclusive"] else "Provisional"
            st.write(
                f"{entry['crop']} — {entry['disease']} "
                f"({entry['confidence']:.1%}; {status})"
            )
            st.caption(entry["timestamp"])


# --------------------------------------------------
# Build the English advisory
# --------------------------------------------------

def build_farmer_report(advisory):
    """Build the complete English report used as translation input."""

    crop, disease = parse_class_name(
        advisory["predicted_class"]
    )

    confidence = advisory["classifier_confidence"] * 100

    lines = [
        "AGRICULTURAL ADVISORY",
        "",
        f"Classifier crop prediction: {crop}",
        f"Classifier disease prediction: {disease}",
        f"Classifier confidence: {confidence:.2f}%",
        "",
        "VISUAL SYMPTOM ANALYSIS",
        advisory.get(
            "visual_analysis",
            "Visual analysis is not available.",
        ),
        "",
        "RETRIEVED AGRICULTURAL KNOWLEDGE",
    ]

    records = advisory.get("retrieved_knowledge", [])

    if records:
        for record in records:
            lines.append(
                f"Crop: {record.get('crop', 'Unknown')}"
            )
            lines.append(
                f"Disease: {record.get('disease', 'Unknown')}"
            )

            information = record.get("text", "")
            if information:
                lines.append(f"Information: {information}")

            score = record.get("score")
            if score is not None:
                lines.append(
                    f"Retrieval similarity: {float(score):.3f}"
                )

            lines.append("")
    else:
        lines.append(
            "No matching knowledge record was retrieved. "
            "This does not mean the disease is absent."
        )

    lines.extend(
        [
            "IMPORTANT LIMITATION",
            advisory.get(
                "notice",
                "This AI-assisted assessment is not a confirmed diagnosis.",
            ),
            "",
            "RECOMMENDED NEXT STEPS",
            "- Inspect several leaves and the whole plant if possible.",
            "- Compare symptoms with reliable local agricultural guidance.",
            "- Consult an agricultural expert if symptoms persist "
            "or the plant is seriously affected.",
            "- Do not apply pesticides solely on the basis of this "
            "AI-generated assessment.",
        ]
    )

    return "\n".join(lines)


# --------------------------------------------------
# Translation and validation
# --------------------------------------------------

def get_display_report(english_report, language):
    """
    Translate with Gemini.

    English remains the source of truth. If translation fails
    or basic checks fail, return English with a warning message.
    """

    if language == "English":
        return english_report, None, False

    try:
        translated = translate_text(
            english_report,
            language,
        )

        passed, reason = translation_quality_check(
            translated,
            language,
        )

        if not passed:
            return english_report, reason, True

        return translated, None, False

    except GeminiTranslationError as error:
        return (
            english_report,
            f"Gemini translation failed: {error}",
            True,
        )

    except Exception as error:
        return (
            english_report,
            f"Unexpected translation error: {type(error).__name__}.",
            True,
        )


# --------------------------------------------------
# Application header
# --------------------------------------------------

st.title("AgriVision AI")

st.caption(
    "Plant leaf classification, visual symptom analysis, "
    "and agricultural knowledge retrieval"
)

st.warning(
    "This prototype supports six trained classes across four crops. "
    "Its predictions are not confirmed diagnoses. High classifier "
    "confidence does not guarantee correctness."
)


# --------------------------------------------------
# Initialize models and knowledge resources
# --------------------------------------------------

try:
    model, class_names = load_classifier()
    embedding_model, index, documents = load_rag()

except Exception as error:
    st.error(
        f"Could not load application resources: {error}"
    )
    st.stop()

st.session_state.setdefault("agri_prediction_history", [])


# --------------------------------------------------
# Sidebar settings
# --------------------------------------------------

with st.sidebar:
    st.header("Assessment settings")

    selected_crop = st.selectbox(
        "Which crop did you upload?",
        SUPPORTED_CROPS,
        help=(
            "Choose the crop shown in the image if known. "
            "Select Unknown if you are unsure."
        ),
    )

    selected_language = st.selectbox(
        "Advisory language",
        LANGUAGES,
        index=0,
    )

    st.caption(
        "Kannada and Hindi translations use Gemini AI. "
        "Translation quality is not guaranteed. Verify important "
        "agricultural decisions with a qualified local expert."
    )

    st.divider()
    if st.button(
        "Clear prediction history",
        key="clear_prediction_history",
    ):
        st.session_state["agri_prediction_history"] = []

    history_container = st.sidebar.empty()
    show_prediction_history(history_container)


# --------------------------------------------------
# Disease knowledge questions
# --------------------------------------------------

st.subheader("Ask a plant-disease question")
st.caption(
    "Searches the indexed agricultural knowledge records. "
    "Retrieved information is reference material, not a diagnosis."
)

with st.form("disease_question_form"):
    disease_question = st.text_input(
        "What would you like to know?",
        placeholder="e.g., What are the symptoms of tomato early blight?",
    )
    question_submitted = st.form_submit_button("Search knowledge")

if question_submitted:
    question = disease_question.strip()
    st.session_state.pop("disease_question_results", None)
    st.session_state.pop("disease_question_error", None)
    st.session_state["disease_question_text"] = question

    if not question:
        st.warning("Enter a question to search the disease knowledge base.")
    elif not documents:
        st.session_state["disease_question_error"] = (
            "The disease knowledge base contains no records."
        )
    else:
        try:
            st.session_state["disease_question_results"] = retrieve(
                question,
                embedding_model,
                index,
                documents,
                top_k=min(3, len(documents)),
            )
        except Exception as error:
            st.session_state["disease_question_error"] = str(error)

question_error = st.session_state.get("disease_question_error")
if question_error:
    st.error(f"Could not search disease knowledge: {question_error}")

question_results = st.session_state.get("disease_question_results")
if question_results is not None:
    st.markdown(
        f"Results for: **{st.session_state['disease_question_text']}**"
    )

    if not question_results:
        st.info("No matching disease knowledge records were found.")

    for result in question_results:
        crop = result.get("crop", "Unknown")
        disease = result.get("disease", "Unknown")
        class_name = result.get("class_name", "Unknown record")

        with st.expander(f"{crop} — {disease}"):
            st.text(result.get("text", "No information is available."))
            st.caption(
                f"Source: knowledge/plant_diseases.json — {class_name}"
            )
            st.caption(
                "Retrieval similarity: "
                f"{float(result.get('score', 0.0)):.3f} "
                "(not a disease probability or diagnosis confidence)."
            )


# --------------------------------------------------
# Upload and display image
# --------------------------------------------------

uploaded_file = st.file_uploader(
    "Upload a clear plant leaf image",
    type=["jpg", "jpeg", "png", "webp"],
)

if uploaded_file is None:
    st.info("Upload a leaf image to begin.")
    st.stop()


try:
    image_bytes = uploaded_file.getvalue()

    from io import BytesIO

    image = Image.open(
        BytesIO(image_bytes)
    ).convert("RGB")

except Exception as error:
    st.error(f"Could not open the uploaded image: {error}")
    st.stop()


st.subheader("Uploaded image")

st.image(
    image,
    caption="Uploaded leaf",
    width=400,
)


# --------------------------------------------------
# Run classifier
# --------------------------------------------------

predictions = predict_image(
    image,
    model,
    class_names,
)

best_prediction = predictions[0]

predicted_crop, predicted_disease = parse_class_name(
    best_prediction["class_name"]
)

confidence = best_prediction["confidence"]
crop_mismatch = (
    selected_crop != "Unknown / not sure"
    and selected_crop.lower() != predicted_crop.lower()
)
inconclusive = is_inconclusive(
    confidence,
    crop_mismatch=crop_mismatch,
)

prediction_history_key = (
    hashlib.sha256(image_bytes).hexdigest(),
    selected_crop,
)
if st.session_state.get("agri_last_history_key") != prediction_history_key:
    st.session_state["agri_prediction_history"].append(
        {
            "crop": predicted_crop,
            "disease": predicted_disease,
            "confidence": confidence,
            "inconclusive": inconclusive,
            "timestamp": datetime.now().astimezone().strftime(
                "%Y-%m-%d %H:%M:%S %Z"
            ),
        }
    )
    st.session_state["agri_last_history_key"] = prediction_history_key
show_prediction_history(history_container)


st.subheader("Classifier results")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Predicted crop",
        predicted_crop,
    )

with col2:
    st.metric(
        "Predicted class",
        predicted_disease,
    )

with col3:
    st.metric(
        "Classifier confidence",
        f"{confidence * 100:.2f}%",
    )

st.caption(
    "Classifier confidence is an uncalibrated softmax score, "
    "not the probability that the prediction is correct."
)


# --------------------------------------------------
# Crop consistency warning
# --------------------------------------------------

if selected_crop != "Unknown / not sure":

    if crop_mismatch:
        st.error(
            f"Crop mismatch: you selected {selected_crop}, but "
            f"the classifier predicted {predicted_crop}. "
            "The prediction may be unreliable. Verify the crop "
            "before using this result."
        )

    else:
        st.info(
            "The selected crop matches the classifier prediction. "
            "This agreement does not confirm the diagnosis."
        )

else:
    st.warning(
        "The crop has not been independently identified. "
        "This classifier supports only Apple, Grape, Potato, "
        "and Tomato classes. It will still assign a class to "
        "unsupported images, so verify the crop."
    )


# --------------------------------------------------
# Confidence explanation
# --------------------------------------------------

if inconclusive:
    st.error(
        "Inconclusive result: "
        + (
            "the selected crop does not match the classifier prediction. "
            if crop_mismatch
            else
            f"the top-class score is below the "
            f"{INCONCLUSIVE_CONFIDENCE_THRESHOLD:.0%} review threshold. "
        )
        + "Do not rely on this disease label; verify the crop and seek "
        "additional evidence. No disease-specific advisory will be generated."
    )

elif confidence >= HIGH_CONFIDENCE_THRESHOLD:
    st.info(
        "The model assigns high confidence to this class. "
        "This is not a calibrated probability that the prediction "
        "is correct."
    )

elif confidence >= INCONCLUSIVE_CONFIDENCE_THRESHOLD:
    st.warning(
        "The model's class confidence is moderate. "
        "Verify the result using additional evidence."
    )

else:
    st.warning(
        "The model's class confidence is low. "
        "Do not rely on this prediction alone."
    )


with st.expander("View top predictions"):
    for item in predictions:
        crop_name, disease_name = parse_class_name(
            item["class_name"]
        )

        label = f"{crop_name} — {disease_name}"

        st.write(
            f"{label}: {item['confidence'] * 100:.2f}%"
        )

        st.progress(
            max(0.0, min(1.0, item["confidence"]))
        )


# --------------------------------------------------
# Generate advisory on demand
# --------------------------------------------------

st.divider()

if st.button(
    "Analyze symptoms and generate advisory",
    type="primary",
    use_container_width=True,
    disabled=inconclusive,
):
    advisory_key = (
        image_bytes,
        selected_crop,
        best_prediction["class_name"],
    )

    st.session_state["agri_advisory_key"] = advisory_key
    st.session_state.pop("agri_advisory", None)
    st.session_state.pop("agri_advisory_error", None)

    with st.spinner(
        "Analyzing symptoms and retrieving agricultural knowledge..."
    ):
        try:
            advisory = generate_advisory(
                image=image,
                predicted_class=best_prediction["class_name"],
                confidence=confidence,
                embedding_model=embedding_model,
                index=index,
                documents=documents,
            )

            st.session_state["agri_advisory"] = advisory

        except Exception as error:
            st.session_state["agri_advisory_error"] = str(error)


# --------------------------------------------------
# Display the current advisory
# --------------------------------------------------

current_key = (
    image_bytes,
    selected_crop,
    best_prediction["class_name"],
)

if st.session_state.get("agri_advisory_key") == current_key:

    advisory_error = st.session_state.get(
        "agri_advisory_error"
    )

    if advisory_error:
        st.error(
            f"Could not generate advisory: {advisory_error}"
        )

    advisory = st.session_state.get("agri_advisory")

    if advisory is not None:

        english_report = build_farmer_report(advisory)

        st.subheader("Farmer advisory")

        # Reuse the translated result when the language is unchanged.
        cache_key = (
            current_key,
            selected_language,
        )

        if st.session_state.get("agri_translation_key") != cache_key:
            with st.spinner(
                f"Preparing advisory in {selected_language}..."
            ):
                displayed_report, warning_message, used_fallback = (
                    get_display_report(
                        english_report,
                        selected_language,
                    )
                )

            st.session_state["agri_translation_key"] = cache_key
            st.session_state["agri_translation_report"] = (
                displayed_report
            )
            st.session_state["agri_translation_warning"] = (
                warning_message
            )
            st.session_state["agri_translation_fallback"] = (
                used_fallback
            )

        displayed_report = st.session_state.get(
            "agri_translation_report",
            english_report,
        )

        warning_message = st.session_state.get(
            "agri_translation_warning"
        )

        used_fallback = st.session_state.get(
            "agri_translation_fallback",
            False,
        )

        if warning_message:
            st.warning(
                f"{warning_message} "
                + (
                    "The original English advisory is displayed instead."
                    if used_fallback
                    else ""
                )
            )

        st.markdown(displayed_report)

        with st.expander("View original English advisory"):
            st.text(english_report)

        st.caption(
            "AI-assisted assessment only. Classifier confidence is not "
            "a guarantee of correctness. Retrieval similarity is not "
            "a disease probability. Translation checks cannot guarantee "
            "linguistic accuracy."
        )
