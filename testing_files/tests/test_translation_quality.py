import unittest

from src.translation.quality import translation_quality_check


class TranslationQualityTests(unittest.TestCase):
    def test_english_text_passes_basic_checks(self):
        self.assertEqual(
            translation_quality_check("Inspect the leaves regularly.", "English"),
            (True, ""),
        )

    def test_empty_text_is_rejected(self):
        passed, reason = translation_quality_check("  ", "Hindi")
        self.assertFalse(passed)
        self.assertIn("empty", reason)

    def test_repetitive_text_is_rejected(self):
        passed, reason = translation_quality_check("repeat " * 35, "English")
        self.assertFalse(passed)
        self.assertIn("repetition", reason)

    def test_missing_expected_script_is_rejected(self):
        for language in ("Kannada", "Hindi"):
            with self.subTest(language=language):
                passed, reason = translation_quality_check(
                    "The leaves should be inspected.",
                    language,
                )
                self.assertFalse(passed)
                self.assertIn(language, reason)

    def test_kannada_script_passes_script_presence_check(self):
        self.assertEqual(
            translation_quality_check("ಎಲೆಗಳನ್ನು ಪರಿಶೀಲಿಸಿ.", "Kannada"),
            (True, ""),
        )

    def test_devanagari_script_passes_script_presence_check(self):
        self.assertEqual(
            translation_quality_check("पत्तियों की जाँच करें।", "Hindi"),
            (True, ""),
        )


if __name__ == "__main__":
    unittest.main()
