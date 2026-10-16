# Testing files

Testing code, test-only data, and generated test/evaluation outputs belong here
so they stay separate from application code and training data.

## Layout

- `tests\` — automated unit tests
- `test_vision.py` — manual Gemma vision smoke test
- `data\plantvillage\splits\color_test.txt` — held-out split manifest
- `data\processed\test\` — held-out PlantVillage test split
- `evaluation\` — evaluation runner, reports, metrics, and confusion matrices

The training split remains in `data\processed\train\`. Dataset preparation
writes the held-out split here automatically.

## Commands (run from the project root)

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s testing_files\tests -v
.\.venv\Scripts\python.exe testing_files\evaluation\run_phase7_evaluation.py
.\.venv\Scripts\python.exe testing_files\evaluation\run_phase7_evaluation.py --external-dir path\to\external_set
.\.venv\Scripts\python.exe testing_files\test_vision.py
```

Evaluation reports and generated results are saved in
`testing_files\evaluation\`. The optional external dataset is read from its
provided path and is not copied into the project.
