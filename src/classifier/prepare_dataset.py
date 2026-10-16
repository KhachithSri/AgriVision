from pathlib import Path
from zipfile import ZipFile
import shutil


# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]

ZIP_PATH = PROJECT_ROOT / "data" / "plantvillage" / "data.zip"
TRAIN_SPLIT = PROJECT_ROOT / "data" / "plantvillage" / "splits" / "color_train.txt"
TEST_SPLIT = (
    PROJECT_ROOT
    / "testing_files"
    / "data"
    / "plantvillage"
    / "splits"
    / "color_test.txt"
)

TRAIN_OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "train"
TEST_OUTPUT_DIR = (
    PROJECT_ROOT / "testing_files" / "data" / "processed" / "test"
)


# Our selected 6 classes
SELECTED_CLASSES = [
    "Apple___Apple_scab",
    "Grape___Black_rot",
    "Potato___Early_blight",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___healthy",
]


def read_split_file(split_path):
    """Read official PlantVillage split file."""
    with open(split_path, "r", encoding="utf-8") as f:
        return [line.strip().replace("\\", "/") for line in f if line.strip()]


def prepare_split(zip_file, split_paths, split_name):
    """Extract only our selected classes for one split."""

    if split_name == "train":
        output_split = TRAIN_OUTPUT_DIR
    elif split_name == "test":
        output_split = TEST_OUTPUT_DIR
    else:
        raise ValueError(f"Unsupported dataset split: {split_name}")

    # Start clean for this split
    if output_split.exists():
        shutil.rmtree(output_split)

    output_split.mkdir(parents=True, exist_ok=True)

    extracted = {class_name: 0 for class_name in SELECTED_CLASSES}

    for zip_path in split_paths:
        # We only want:
        # raw/color/<selected_class>/<image>
        parts = zip_path.split("/")

        if len(parts) < 4:
            continue

        class_name = parts[2]

        if class_name not in SELECTED_CLASSES:
            continue

        output_class_dir = output_split / class_name
        output_class_dir.mkdir(parents=True, exist_ok=True)

        output_file = output_class_dir / parts[-1]

        with zip_file.open(zip_path) as source:
            with open(output_file, "wb") as destination:
                shutil.copyfileobj(source, destination)

        extracted[class_name] += 1

    print(f"\n{split_name.upper()} extraction complete:")

    total = 0

    for class_name, count in extracted.items():
        print(f"  {class_name}: {count} images")
        total += count

    print(f"  TOTAL: {total} images")


def main():
    print("Checking required files...")

    if not ZIP_PATH.exists():
        raise FileNotFoundError(f"Missing dataset ZIP: {ZIP_PATH}")

    if not TRAIN_SPLIT.exists():
        raise FileNotFoundError(f"Missing train split: {TRAIN_SPLIT}")

    if not TEST_SPLIT.exists():
        raise FileNotFoundError(f"Missing test split: {TEST_SPLIT}")

    print("All required files found.")

    print("\nReading official PlantVillage train/test splits...")

    train_paths = read_split_file(TRAIN_SPLIT)
    test_paths = read_split_file(TEST_SPLIT)

    print(f"Train split entries: {len(train_paths)}")
    print(f"Test split entries:  {len(test_paths)}")

    print("\nOpening PlantVillage ZIP...")
    with ZipFile(ZIP_PATH, "r") as zip_file:

        print("\nPreparing TRAIN dataset...")
        prepare_split(zip_file, train_paths, "train")

        print("\nPreparing TEST dataset...")
        prepare_split(zip_file, test_paths, "test")

    print("\nDataset preparation finished.")
    print(f"Processed training data: {TRAIN_OUTPUT_DIR}")
    print(f"Processed test data: {TEST_OUTPUT_DIR}")


if __name__ == "__main__":
    main()