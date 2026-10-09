
import json
import sys
from pathlib import Path

import torch
from torch import nn
from torchvision import models, transforms
from PIL import Image

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"
device = torch.device("cpu")

# Check that the model files exist
model_path = MODEL_DIR / "brain_tumor_resnet18.pth"
classes_path = MODEL_DIR / "classes.json"

if not model_path.exists() or not classes_path.exists():
    raise FileNotFoundError(
        "Model files not found. Run train.py first."
    )

with open(classes_path, "r") as file:
    classes = json.load(file)

# Re-create the same architecture used during training
model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, len(classes))

model.load_state_dict(
    torch.load(
        model_path,
        map_location=device,
        weights_only=True,
    )
)
model.eval()

# Use the same preprocessing as the test pipeline
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


def predict_image(image_path):
    image_path = Path(image_path)

    if not image_path.is_file():
        raise FileNotFoundError(
            f"Image not found: {image_path}"
        )

    with Image.open(image_path) as image:
        image = image.convert("RGB")
        tensor = preprocess(image).unsqueeze(0)

    with torch.no_grad():
        logits = model(tensor)
        scores = torch.softmax(logits, dim=1)[0]

    predicted_index = int(scores.argmax())

    print("\nPredicted dataset class:",
          classes[predicted_index])
    print(f"Model score: {scores[predicted_index].item():.2%}")

    print("\nScores for all classes:")
    for name, score in zip(classes, scores.tolist()):
        print(f"{name}: {score:.2%}")

    print("\nEducational research prototype only.")
    print("Not intended for medical diagnosis.")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(
            'Usage: python predict.py "path/to/mri_image.jpg"'
        )
        sys.exit(1)

    predict_image(sys.argv[1])

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print('Usage: python predict.py "path/to/mri_image.jpg"')
        sys.exit(1)

    predict_image(sys.argv[1])
