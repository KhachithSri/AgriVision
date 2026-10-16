def translation_quality_check(text, language=None):
    """Reject empty, obviously repetitive, or wrong-script translations."""

    if not isinstance(text, str) or not text.strip():
        return False, "The translation was empty."

    words = text.lower().split()

    if len(words) >= 30:
        unique_ratio = len(set(words)) / len(words)

        if unique_ratio < 0.25:
            return False, (
                "The translation contains excessive repetition."
            )

    suspicious_phrases = [
        "cultivators advisory",
        "cultivators cultivators",
    ]

    lower_text = text.lower()

    if any(
        phrase in lower_text
        for phrase in suspicious_phrases
    ):
        return False, (
            "The translation contains malformed repeated text."
        )

    script_ranges = {
        "Kannada": (0x0C80, 0x0CFF),
        "Hindi": (0x0900, 0x097F),
    }
    script_range = script_ranges.get(language)
    if script_range and not any(
        script_range[0] <= ord(character) <= script_range[1]
        for character in text
    ):
        return False, (
            f"The translation does not contain {language} script."
        )

    return True, ""
