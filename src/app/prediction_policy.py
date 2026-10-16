INCONCLUSIVE_CONFIDENCE_THRESHOLD = 0.60
HIGH_CONFIDENCE_THRESHOLD = 0.85


def is_inconclusive(confidence, crop_mismatch=False):
    """Return whether the UI should withhold a disease-specific advisory."""

    if not 0.0 <= confidence <= 1.0:
        raise ValueError("Classifier confidence must be between 0 and 1.")

    return (
        confidence < INCONCLUSIVE_CONFIDENCE_THRESHOLD
        or crop_mismatch
    )
