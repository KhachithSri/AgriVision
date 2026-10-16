# 8/10/26

## created virtual environment for roject execution.

```powershell
PS D:\AgriVision-AI> python -m venv .venv
PS D:\AgriVision-AI> .venv\Scripts\Activate.ps1
```

## created project framework

```text
AgriVision-AI
│
├── data
├── models
├── knowledge
├── evaluation
│
├── src
│   ├── classifier
│   ├── rag
│   ├── llm
│   └── app
```

## installed some packages

```powershell
pip install torch torchvision
pip install transformers datasets pillow opencv-python scikit-learn matplotlib seaborn
pip install streamlit faiss-cpu sentence-transformers
pip install python-dotenv
```

created requirements.txt and excecuted - 

```powershell
pip freeze > requirements.txt
```

## Create a dataset inspection file - inspect_dataset.py then ran python src\classifier\inspect_dataset.py

## Download the PlantVillage dataset

Because you have 8 GB RAM, we should avoid loading the whole dataset into memory.

We will:

```text
Hugging Face PlantVillage
        ↓
Download data.zip
        ↓
Extract locally
        ↓
Inspect disease classes
        ↓
Select 3–5 classes
```

## Install the Hugging Face download utility - pip install huggingface_hub

## Created a download script src\classifier\download_dataset.py and ran python src\classifier\download_dataset.py which downloaded dataset in zip format

## Created src\classifier\inspect_zip.py and ran - python src\classifier\inspect_zip.py which simply looks inside:

```text
data/plantvillage/data.zip
```

## Before extraction, let's verify the six classes- Create src\classifier\check_selected_classes.py and ran - python src\classifier\check_selected_classes.py

## Create the extraction script- src\classifier\prepare_dataset.py and ran it

finished extraction of dataset

## created src\classifier\train_classifier.py and ran src\classifier\train_classifier.py

## created backup for train_classifier.py as train_classifier_initial.py and replaced new code in train_classifier.py to create a proper training/validation/test split

## created src/classifier/predict_image.py and ran -  python src\classifier\predict_image.py "D:\AgriVision-AI\testing_files\data\processed\test\Tomato___Early_blight\00c5c908-fc25-4710-a109-db143da23112___RS_Erly.B 7778.JPG" --- got 92.65% accuracy

# 13/10/26

## building RAG and FAISS

## created disease knowledge file- knowledge\plant_diseases.json

## Created the FAISS knowledge index src\rag\build_index.py and ran - python src\rag\build_index.py

## Created the RAG retrieval script src\rag\retrieve.py and ran - python src\rag\retrieve.py

## Building the classifier + RAG integration: 

## Created the integration script src\app\agri_pipeline.py and ran - python src\app\agri_pipeline.py

## The AgriVision AI pipeline is working end-to-end. 

The image classifier correctly predicted Tomato Early Blight, and the RAG component retrieved the matching disease information.

### Current project status

| Item | Status |
|---|---|
| PlantVillage dataset preparation | Completed |
| MobileNetV3-Small training and evaluation | Completed |
| Single-image disease prediction | Completed |
| FAISS knowledge index construction | Completed |
| Semantic retrieval testing | Completed |
| Classifier + RAG integration | Completed |

### Verified prediction

| Item | Result |
|---|---|
| Predicted disease | Tomato Early Blight |
| Model confidence | 92.65% |

## Created the Streamlit application src\app\streamlit_app.py and ran - python -m streamlit run src\app\streamlit_app.py

## edited the code in src\app\streamlit_app.py to add feature of question and answering using FAISS

this completes phase 1 and 2 of our flowplan

## phase 3

### Model 1: Our trained MobileNetV3

Already completed

- Predicts one of our six disease classes.
- Produces a classification confidence score.
- Has achieved 97.61% test accuracy on our current dataset.

### Model 2: Gemma 3 4B or Gemini

- Examines the image and describes visible symptoms in natural language.
- Helps explain observations to the farmer.
- Supports the multimodal GenAI requirement in your professor's document.

## installed ollama for windows

## downloaded the model - ollama pull gemma3:4b

## created files src\vision\__init__.py and src\vision\symptom_analyzer.py

## created file testing_files\test_vision.py and tested gemma

## updated streamlit_app.py with proper reply

## implementing multilanguage in file src\app\streamlit_app.py

## local llm translation was not working properly, so integrated gemini api key

created a folder translation and files init.py and gemini_translator.py

verified the translation..

## integrating translation to steamlit app

updated streamlit_app.py

## performed testing of pipelines

## organized the test data separately in testing_files folder.
