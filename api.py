
import json
from io import BytesIO
from pathlib import Path

import torch
from torch import nn
from torchvision import models, transforms
from PIL import Image, UnidentifiedImageError
from fastapi import FastAPI, File, HTTPException, UploadFile

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"
DEVICE = torch.device("cpu")

app = FastAPI(
    title="Brain MRI Classification API",
    description="Educational PyTorch image-classification prototype.",
    version="1.0.0",
)

# Load the model once when the API starts
def load_model():
    model_path = MODEL_DIR / "brain_tumor_resnet18.pth"
    classes_path = MODEL_DIR / "classes.json"

    if not model_path.exists() or not classes_path.exists():
        raise FileNotFoundError(
            "Model files are missing. Run train.py first."
        )

    with open(classes_path, "r") as file:
        classes = json.load(file)

    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(classes))
    model.load_state_dict(
        torch.load(
            model_path,
            map_location=DEVICE,
            weights_only=True,
        )
    )
    model.eval()

    return model, classes


model, class_names = load_model()

preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


@app.get("/")
def home():
    return {
        "message": "Brain MRI Classification API",
        "docs": "/docs",
        "purpose": "Educational research only; not for diagnosis.",
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": True,
        "device": str(DEVICE),
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if file.content_type not in {
        "image/jpeg", "image/png", "image/webp", "image/bmp"
    }:
        raise HTTPException(
            status_code=415,
            detail="Upload a JPEG, PNG, WEBP, or BMP image.",
        )

    # Limit uploads to 10 MB
    contents = await file.read(10 * 1024 * 1024 + 1)

    if len(contents) > 10 * 1024 * 1024:
        raise HTTPException(
            status_code=413,
            detail="Image must be 10 MB or smaller.",
        )

    try:
        with Image.open(BytesIO(contents)) as image:
            image = image.convert("RGB")
            tensor = preprocess(image).unsqueeze(0)
    except (UnidentifiedImageError, OSError, ValueError):
        raise HTTPException(
            status_code=400,
            detail="The uploaded file is not a valid image.",
        )

    with torch.no_grad():
        logits = model(tensor.to(DEVICE))
        probabilities = torch.softmax(logits, dim=1)[0]

    index = int(probabilities.argmax())

    return {
        "predicted_class": class_names[index],
        "model_score": round(probabilities[index].item(), 6),
        "scores": {
            name: round(score, 6)
            for name, score in zip(
                class_names, probabilities.tolist()
            )
        },
        "device": str(DEVICE),
        "disclaimer": (
            "Educational research prototype only; "
            "not for medical diagnosis."
        ),
    }
