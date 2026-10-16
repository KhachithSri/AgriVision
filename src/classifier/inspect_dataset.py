from huggingface_hub import HfApi

dataset_name = "mohanty/PlantVillage"

print("Checking PlantVillage repository...")
print()

api = HfApi()

info = api.dataset_info(dataset_name)

print("Dataset:", info.id)
print("Author:", info.author)
print("Last modified:", info.lastModified)

print("\nAvailable files:")
for file in info.siblings:
    print(file.rfilename)