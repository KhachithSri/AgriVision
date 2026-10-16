from huggingface_hub import hf_hub_download
from pathlib import Path

dataset_dir = Path("data/plantvillage")
dataset_dir.mkdir(parents=True, exist_ok=True)

print("Downloading PlantVillage data.zip...")
print("This may take some time because the dataset is large.")

zip_path = hf_hub_download(
    repo_id="mohanty/PlantVillage",
    filename="data.zip",
    repo_type="dataset",
    local_dir=str(dataset_dir)
)

print("\nDownload completed.")
print("File:", zip_path)