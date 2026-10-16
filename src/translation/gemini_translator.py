
import os
import time

from dotenv import load_dotenv
from google import genai

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)
ENV_FILE = os.path.join(PROJECT_ROOT, ".env")
load_dotenv(ENV_FILE)

MODEL_NAME = os.getenv(
    "GEMINI_TRANSLATION_MODEL",
    "gemini-3.8-flash",
)

LANGUAGE_NAMES = {
    "English": "English",
    "Kannada": "Kannada",
    "Hindi": "Hindi",
}


class GeminiTranslationError(RuntimeError):
    """Raised when Gemini translation cannot be completed."""


def translate_text(text: str, language: str) -> str:
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Text to translate cannot be empty.")

    if language not in LANGUAGE_NAMES:
        raise ValueError(
            f"Unsupported language: {language}. "
            f"Choose from {list(LANGUAGE_NAMES)}."
        )

    if language == "English":
        return text

    api_key = os.getenv("GEMINI_API_KEY", "")
    if not api_key or api_key == "PASTE_YOUR_ACTUAL_API_KEY_HERE":
        raise GeminiTranslationError(
            "GEMINI_API_KEY is missing from the project-root .env file."
        )

    prompt = f"""You are a professional translator for agricultural
information intended for farmers in India.

Translate the complete source text into natural, fluent,
grammatically correct {LANGUAGE_NAMES[language]}.

Requirements:
1. Use natural {LANGUAGE_NAMES[language]} throughout.
2. For Kannada, use Kannada script. For Hindi, use Devanagari.
3. Avoid unnecessary English words and language mixing.
4. Preserve headings, bullet points, numbers and important facts.
5. Preserve uncertainty, warnings and limitations.
6. Never turn a possible disease into a confirmed diagnosis.
7. Do not add or invent agricultural recommendations.
8. Retain scientific names in English when necessary.
9. Translate the entire text without summarizing.
10. Return only the translation.

SOURCE TEXT:
{text}
"""

    client = genai.Client(api_key=api_key)
    last_error = None

    # Three attempts maximum; wait 2 seconds, then 4 seconds.
    for attempt in range(3):
        try:
            response = client.models.generate_content(
                model=MODEL_NAME,
                contents=prompt,
                config={
                    "temperature": 0.1,
                    "max_output_tokens": 4000,
                },
            )

            translated = (response.text or "").strip()
            if not translated:
                raise GeminiTranslationError(
                    "Gemini returned an empty translation."
                )

            return translated

        except Exception as error:
            details = str(error)
            if api_key:
                details = details.replace(api_key, "[REDACTED]")

            last_error = (
                f"{type(error).__name__}: {details[:600]}"
            )

            # Retry only likely temporary server/rate-limit failures.
            retryable = any(
                marker in details.upper()
                for marker in (
                    "503",
                    "UNAVAILABLE",
                    "500",
                    "INTERNAL",
                    "429",
                    "RESOURCE_EXHAUSTED",
                )
            )

            if not retryable or attempt == 2:
                break

            time.sleep(2 ** (attempt + 1))

    raise GeminiTranslationError(
        f"Translation failed after retries or a non-retryable error. "
        f"{last_error}"
    ) from None
