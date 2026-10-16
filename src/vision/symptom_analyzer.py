import base64
import io
import requests
from PIL import Image

OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL_NAME = "gemma3:4b"
TIMEOUT_SECONDS = 300


def _ollama_chat(messages, temperature=0.1, num_predict=400):
    """Send a request to the local Ollama model."""

    response = requests.post(
        OLLAMA_URL,
        json={
            "model": MODEL_NAME,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": num_predict,
            },
        },
        timeout=TIMEOUT_SECONDS,
    )

    response.raise_for_status()
    result = response.json()

    return result["message"]["content"].strip()


def analyze_leaf_image(
    image: Image.Image,
    expected_class: str = "",
) -> str:
    """Describe visible leaf symptoms without confirming a diagnosis."""

    image_buffer = io.BytesIO()
    image.convert("RGB").save(image_buffer, format="JPEG")

    image_base64 = base64.b64encode(
        image_buffer.getvalue()
    ).decode("utf-8")

    prompt = f"""
Analyze this plant leaf image for agricultural assistance.

Classifier's provisional prediction: {expected_class or "Not provided"}

Rules:
- Describe only visible features supported by the image.
- Do not automatically agree with the classifier.
- State when the image is unclear or insufficient.
- Do not claim a confirmed disease diagnosis.
- Do not invent symptoms.
- Recommend practical, low-risk next steps.

Use these headings:
VISIBLE SYMPTOMS
CONSISTENCY CHECK
POSSIBLE EXPLANATION
NEXT STEP

Keep the response under 150 words.
"""

    return _ollama_chat(
        messages=[
            {
                "role": "user",
                "content": prompt,
                "images": [image_base64],
            }
        ],
        temperature=0.1,
        num_predict=300,
    )


def translate_advisory(text: str, language: str) -> str:
    """Translate an advisory while preserving meaning and uncertainty."""

    language_names = {
        "English": "English",
        "Kannada": "Kannada",
        "Hindi": "Hindi",
    }

    if language not in language_names:
        raise ValueError(
            f"Unsupported language: {language}. "
            f"Choose one of {list(language_names)}."
        )

    if not text or not text.strip():
        return ""

    if language == "English":
        return text

    prompt = f"""
You are a professional agricultural translator for farmers in India.

Translate the advisory below into natural, grammatically correct
{language_names[language]}.

STRICT RULES:
1. Write headings and explanations in {language_names[language]}.
2. Do not mix Kannada, Hindi, and English sentences.
3. Retain scientific names and necessary technical terms where appropriate.
4. Preserve all facts, warnings, and uncertainty.
5. Do not add diagnoses, treatments, or recommendations.
6. Never change a possible disease into a confirmed diagnosis.
7. Keep the original sections and approximately the same length.
8. Return only the translated advisory.

For Kannada, use natural Kannada script and grammar.
For Hindi, use natural Devanagari script and grammar.

ADVISORY:
{text}
"""

    translated = _ollama_chat(
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        temperature=0.0,
        num_predict=500,
    )

    if not translated:
        raise RuntimeError("The model returned an empty translation.")

    return translated
