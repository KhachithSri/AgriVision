# AgriVision AI Action Log

This file records actions and commands performed while completing the Phase 1–7 carry-over work.

## 2026-10-16

| Action / command | Result |
|---|---|
| Inspected the phase checklist, Streamlit app, RAG/advisory flow, Gemma vision module, translation module, evaluation code/results, and project tests. | Identified missing prediction history, inconclusive handling, language-output checks, and outstanding external evaluation/reviewer requirements. |
| Checked the project `.venv`, dataset splits, Ollama model/API, Gemini-key configuration presence (without reading or displaying its value), and test files. | Project dependencies and local `gemma3:4b` were available. No separate external/field-image evaluation set was present. |
| Added `src/app/prediction_policy.py` and updated `src/app/streamlit_app.py`. | Added session-only prediction history and clear-history control. Scores below 0.60 and selected-crop mismatches now show an inconclusive state and block disease-specific advisory generation. |
| Updated `testing_files/evaluation/run_phase7_evaluation.py` and `testing_files/evaluation/PHASE7_EVALUATION_REPORT.md`. | Evaluation now reports held-out accuracy by classifier-score band and documents the results as in-distribution benchmarks, not calibrated correctness probabilities. |
| Added `src/translation/quality.py` and `testing_files/tests/test_translation_quality.py`. | Added checks for empty/repetitive translation output and missing Kannada/Devanagari script. These checks do not establish fluency or factual accuracy. |
| Added `testing_files/tests/test_prediction_policy.py`. | Added tests for the review threshold, crop mismatch, and invalid score range. |
| `.venv\Scripts\python.exe -m unittest discover -s tests -v` | All 10 focused unit tests passed. |
| `.venv\Scripts\python.exe -m py_compile src\app\streamlit_app.py src\app\prediction_policy.py src\translation\quality.py testing_files\evaluation\run_phase7_evaluation.py testing_files\tests\test_prediction_policy.py testing_files\tests\test_translation_quality.py` | Python compilation passed. |
| `.venv\Scripts\python.exe testing_files\evaluation\run_phase7_evaluation.py` | Held-out evaluation completed on 1,379 PlantVillage images: accuracy 97.61%, macro F1 97.84%. Updated `testing_files/evaluation/held_out_test_evaluation.json` with score bands. |
| Ran a Gemma/RAG advisory smoke test using one held-out Tomato Early blight image. | First assertion incorrectly expected `class_name` in the advisory result. Corrected the check to use the returned crop/disease fields; rerun produced visual text, the matching Tomato / Early blight record, and advisory notice. |
| `.venv\Scripts\python.exe -m streamlit run src\app\streamlit_app.py --server.headless true --server.port 8765` | Started the app for a UI smoke test. |
| `curl.exe -sS -I --max-time 10 http://localhost:8765` | Received HTTP 200. Inspected the page: disease-question form and prediction history were visible without startup errors. |
| `Stop-Process -Id 6600` followed by a `curl.exe` probe of port 8765 | Stopped the temporary smoke-test server; later probe confirmed the port was closed. |
| Ran Pylance diagnostics on changed app, evaluation, and test files. | No diagnostics reported. |
| Checked for external-image and human review evidence. | No separately sourced, labelled field-image set or qualified reviewer was available. |
| Checked whether a live Gemini translation smoke test was authorized. | No explicit authorization was received; no Gemini API request was made. |
| 2026-10-16 follow-up: inspected `data` subdirectories and `evaluation` artifacts; asked for the external dataset location and reviewer scores. | Confirmed only PlantVillage `train`/`test` data is in the workspace. User confirmed the external dataset is not present; no reviewer scores or reviewer availability were supplied. |
| Asked whether to create blank Phase 7 intake and reviewer templates. | User chose not to create templates. No templates were added. |

## Still outstanding

- Run a live Kannada/Hindi Gemini translation test after explicit API-use authorization; it may incur usage charges.
- Evaluate a separately sourced, labelled field-image set using the documented protocol.
- Complete qualified human review of visual grounding and RAG-advisory faithfulness.
- Phase 7 cannot be signed off until the external labelled images and qualified reviewer ratings are supplied and evaluated.

## Testing-file separation (2026-10-16)

Moved evaluation scripts, reports, metrics and figures to
`testing_files/evaluation/`; automated tests to `testing_files/tests/`;
the Gemma smoke-test script to `testing_files/test_vision.py`; and the
PlantVillage held-out test split to `testing_files/data/processed/test/`.
Updated dataset preparation, classifier training and evaluation paths to keep
test data and outputs separate from application files and training images.
See `testing_files/README.md` for current commands.
Earlier command rows in the historical log above are preserved as originally
run; the commands in `testing_files/README.md` are the current locations.

| Command / check | Result |
|---|---|
| `Move-Item evaluation testing_files\evaluation; Move-Item tests testing_files\tests; Move-Item test_vision.py testing_files\test_vision.py; Move-Item data\processed\test testing_files\data\processed\test` | Relocated evaluation code/results, unit tests, manual vision test, and held-out test images. Updated the affected data-preparation, training, evaluation, and documentation paths. |
| Moved `data\plantvillage\splits\color_test.txt` to `testing_files\data\plantvillage\splits\color_test.txt` and updated `src\classifier\prepare_dataset.py`. | The held-out split manifest now stays with the testing files and dataset-preparation code reads it from the new location. |
| `.venv\Scripts\python.exe -m unittest discover -s testing_files\tests -v` | All 10 tests passed from the relocated test folder. |
| `.venv\Scripts\python.exe -m py_compile src\classifier\prepare_dataset.py src\classifier\train_classifier.py src\classifier\train_classifier_initial.py testing_files\evaluation\run_phase7_evaluation.py testing_files\test_vision.py testing_files\tests\test_prediction_policy.py testing_files\tests\test_translation_quality.py` | Compilation passed. |
| `.venv\Scripts\python.exe testing_files\evaluation\run_phase7_evaluation.py` | Evaluation passed against the relocated test split; 1,379 images, 97.61% accuracy and 97.84% macro F1. Result JSON now records `testing_files\data\processed\test` as its dataset path. |
| Checked old locations `evaluation`, `tests`, `test_vision.py`, and `data\processed\test`. | None remain at the old locations. |
| Verified `src.classifier.prepare_dataset`, `src.classifier.train_classifier`, and `testing_files.evaluation.run_phase7_evaluation` path constants. | Dataset preparation, training, and evaluation resolve the same relocated held-out dataset directory. |

## Project flow Markdown formatting (2026-10-16)

Reformatted `Z-Documentation/pojectflow.md` using Markdown headings, fenced command/tree blocks, lists, and tables. Preserved the original wording and sentences.
