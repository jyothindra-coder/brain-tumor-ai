# Brain MRI Classification Using Deep Learning

An educational computer vision project that classifies brain MRI images into four dataset categories using PyTorch and ResNet18 transfer learning. It includes a Streamlit interface, a FastAPI inference service, model evaluation, and automated API tests.

> **Disclaimer:** This is an educational research prototype, not a medical device. It must not be used for medical diagnosis or treatment decisions.

## Features

- Four-class MRI image classification
- ResNet18 transfer learning with PyTorch
- CPU-based model inference
- Streamlit web interface
- FastAPI prediction endpoint and interactive API documentation
- Training/validation split and held-out test evaluation
- Exact and perceptual-hash duplicate investigation scripts
- Automated API tests using pytest

## Dataset

Classes: `glioma`, `meningioma`, `notumor`, `pituitary`.

Dataset source: [Brain Tumor MRI Dataset on Kaggle](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset)

Place the dataset locally in `data/Training/` and `data/Testing/`, with one folder per class. The dataset is excluded from this Git repository.

## Technology Stack

Python, PyTorch, Torchvision, ResNet18, Pillow, scikit-learn, Streamlit, FastAPI, Uvicorn, pytest, HTTPX, and ImageHash.

## Setup

Use Python 3.11 on Windows PowerShell.

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1

python -m pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
```

Add the dataset to the local `data/` directory before training.

## Run the Project

Train the baseline model:

```powershell
python train.py
```

Train and evaluate version 2:

```powershell
python train_v2.py
```

Launch the Streamlit application:

```powershell
python -m streamlit run app.py
```

Launch the FastAPI service:

```powershell
python -m uvicorn api:app --reload
```

Open `http://127.0.0.1:8000/docs` for interactive API documentation.

Run automated tests:

```powershell
python -m pytest tests -v
```

## Experimental Results

| Experiment | Validation accuracy | Test accuracy |
|---|---:|---:|
| Baseline | Not separately measured | 83.13% |
| Version 2 | 89.29% | 82.75% |

Version 2 selects its best checkpoint using validation accuracy. It did not improve test accuracy over the baseline. The reported metrics describe performance on this dataset and do not establish clinical safety or generalization to other hospitals, scanners, or patient populations.

## Data Quality and Limitations

- No exact duplicate files were reported across the training and testing folders by SHA-256 hashing.
- Perceptual-hash matches, if any, require manual review.
- Patient-level independence has not been established.
- Image-level splitting can allow related images from the same patient to appear in different subsets.
- Glioma recall requires further investigation.
- Model scores are not calibrated medical probabilities.

## Future Work

- Investigate near-duplicates and patient-level overlap.
- Evaluate on an independently sourced dataset.
- Investigate class-specific errors and alternative training strategies.
- Add reproducible configurations, model versioning, and continuous integration.
- Harden the API before public deployment.

## Responsible Use

This project is intended for learning and research only. Its predictions must not be used to diagnose disease, rule out disease, or guide clinical decisions.

## License

Confirm the licensing terms of the code, dataset, and third-party assets before choosing a repository license or redistributing model files.