
import csv
import json
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data" / "Testing"
MODEL_DIR = BASE_DIR / "models"
OUTPUT_DIR = BASE_DIR / "outputs" / "evaluation"

DEVICE = torch.device("cpu")
BATCH_SIZE = 8

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])

dataset = datasets.ImageFolder(DATA_DIR, transform=transform)
loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

models_to_evaluate = [
    {
        "name": "baseline",
        "weights": MODEL_DIR / "brain_tumor_resnet18.pth",
        "classes": MODEL_DIR / "classes.json",
    },
    {
        "name": "v2",
        "weights": MODEL_DIR / "brain_tumor_resnet18_v2.pth",
        "classes": MODEL_DIR / "classes_v2.json",
    },
]

all_results = []

print("Device:", DEVICE)
print("Test images:", len(dataset))
print("Dataset classes:", dataset.classes)


def evaluate_model(config):
    weights_path = config["weights"]
    classes_path = config["classes"]

    if not weights_path.is_file() or not classes_path.is_file():
        raise FileNotFoundError(
            f"Missing model or class file for {config['name']}."
        )

    with open(classes_path, "r", encoding="utf-8") as file:
        class_names = json.load(file)

    if class_names != dataset.classes:
        raise ValueError(
            f"Class order mismatch for {config['name']}: "
            f"model={class_names}, dataset={dataset.classes}"
        )

    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(class_names))

    state = torch.load(
        weights_path,
        map_location=DEVICE,
        weights_only=True,
    )
    model.load_state_dict(state)
    model.to(DEVICE)
    model.eval()

    actual = []
    predicted = []
    records = []

    with torch.inference_mode():
        for images, labels in loader:
            logits = model(images.to(DEVICE))
            probabilities = torch.softmax(logits, dim=1)
            predictions = probabilities.argmax(dim=1)

            actual.extend(labels.tolist())
            predicted.extend(predictions.cpu().tolist())

            for label, prediction, scores in zip(
                labels.tolist(),
                predictions.cpu().tolist(),
                probabilities.cpu().tolist(),
            ):
                image_path, _ = dataset.samples[len(records)]
                records.append({
                    "image": str(Path(image_path).relative_to(BASE_DIR)),
                    "actual": class_names[label],
                    "predicted": class_names[prediction],
                    "correct": label == prediction,
                    "top_score": max(scores),
                    "actual_class_score": scores[label],
                })

    accuracy = accuracy_score(actual, predicted)
    matrix = confusion_matrix(
        actual,
        predicted,
        labels=list(range(len(class_names))),
    )

    report = classification_report(
        actual,
        predicted,
        labels=list(range(len(class_names))),
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )

    print(f"\n{'=' * 55}")
    print("MODEL:", config["name"])
    print(f"Test accuracy: {accuracy:.4%}")
    print("\nClassification report:")
    print(classification_report(
        actual,
        predicted,
        labels=list(range(len(class_names))),
        target_names=class_names,
        zero_division=0,
    ))
    print("Confusion matrix (rows=actual, columns=predicted):")
    print("Class order:", class_names)
    print(matrix)

    prefix = OUTPUT_DIR / config["name"]

    with open(
        f"{prefix}_report.json", "w", encoding="utf-8"
    ) as file:
        json.dump({
            "model": config["name"],
            "test_accuracy": accuracy,
            "classes": class_names,
            "classification_report": report,
            "confusion_matrix": matrix.tolist(),
        }, file, indent=2)

    with open(
        f"{prefix}_predictions.csv", "w",
        newline="", encoding="utf-8-sig"
    ) as file:
        writer = csv.DictWriter(file, fieldnames=records[0].keys())
        writer.writeheader()
        writer.writerows(records)

    all_results.append({
        "model": config["name"],
        "test_accuracy": accuracy,
        "macro_f1": report["macro avg"]["f1-score"],
    })


for config in models_to_evaluate:
    evaluate_model(config)

print(f"\n{'=' * 55}")
print("SUMMARY")
for result in all_results:
    print(
        f"{result['model']}: "
        f"accuracy={result['test_accuracy']:.4%}, "
        f"macro-F1={result['macro_f1']:.4f}"
    )

print("\nSaved evaluation files to:", OUTPUT_DIR)
print("These results use the existing held-out test set.")
