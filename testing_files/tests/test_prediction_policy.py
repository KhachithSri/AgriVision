import unittest

from src.app.prediction_policy import (
    INCONCLUSIVE_CONFIDENCE_THRESHOLD,
    is_inconclusive,
)


class PredictionPolicyTests(unittest.TestCase):
    def test_score_below_review_threshold_is_inconclusive(self):
        self.assertTrue(
            is_inconclusive(INCONCLUSIVE_CONFIDENCE_THRESHOLD - 0.01)
        )

    def test_score_at_review_threshold_is_not_inconclusive(self):
        self.assertFalse(
            is_inconclusive(INCONCLUSIVE_CONFIDENCE_THRESHOLD)
        )

    def test_crop_mismatch_is_inconclusive_at_any_score(self):
        self.assertTrue(is_inconclusive(0.99, crop_mismatch=True))

    def test_out_of_range_score_is_rejected(self):
        for confidence in (-0.01, 1.01):
            with self.subTest(confidence=confidence):
                with self.assertRaises(ValueError):
                    is_inconclusive(confidence)


if __name__ == "__main__":
    unittest.main()
