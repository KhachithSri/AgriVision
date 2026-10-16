
from PIL import Image
from src.vision.symptom_analyzer import analyze_leaf_image

path = input("Enter the full path to a leaf image: ").strip().strip('"')

with Image.open(path) as image:
    print("\nGemma 3 4B is analyzing the image...\n")
    result = analyze_leaf_image(image)

print(result)
