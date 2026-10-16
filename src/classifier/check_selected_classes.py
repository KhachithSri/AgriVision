from pathlib import Path
from zipfile import ZipFile

ZIP_PATH = Path("data/plantvillage/data.zip")

SELECTED_CLASSES = [
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Potato___Early_blight",
    "Grape___Black_rot",
    "Apple___Apple_scab",
    "Tomato___healthy",
]

with ZipFile(ZIP_PATH, "r") as z:
    files = z.namelist()

print("Checking selected classes...\n")

for class_name in SELECTED_CLASSES:
    prefix = f"raw/color/{class_name}/"

    images = [
        file for file in files
        if file.startswith(prefix)
        and file.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    print(f"{class_name}: {len(images)} images")

print("\nClass verification complete.")