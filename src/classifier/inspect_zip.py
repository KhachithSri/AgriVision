from pathlib import Path
from zipfile import ZipFile
from collections import Counter

zip_path = Path("data/plantvillage/data.zip")

print("Opening PlantVillage ZIP...")

with ZipFile(zip_path, "r") as z:
    files = z.namelist()

print(f"\nTotal files in ZIP: {len(files)}")

print("\nFirst 20 files:")
for file in files[:20]:
    print(file)

# Look for image files
image_extensions = (".jpg", ".jpeg", ".png")

image_files = [
    file for file in files
    if file.lower().endswith(image_extensions)
]

print(f"\nTotal image files: {len(image_files)}")

# Look at directory structure
directories = set()

for file in image_files:
    parts = Path(file).parts

    if len(parts) >= 2:
        directories.add(parts[-2])

print("\nPossible disease/class folders:")

for folder in sorted(directories):
    print(folder)

print(f"\nNumber of possible classes: {len(directories)}")