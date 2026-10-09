
from pathlib import Path
import json
import random

import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, models, transforms
from torchvision.models import ResNet18_Weights
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix


# ---------------- Configuration ----------------
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"

TRAIN_DIR = DATA_DIR / "Training"
TEST_DIR = DATA_DIR / "Testing"

MODEL_PATH = MODEL_DIR / "brain_tumor_resnet18_v2.pth"
CLASSES_PATH = MODEL_DIR / "classes_v2.json"

SEED = 42
BATCH_SIZE = 8
EPOCHS = 5
LEARNING_RATE = 0.001
DEVICE = torch.device("cpu")

random.seed(SEED)
torch.manual_seed(SEED)
MODEL_DIR.mkdir(parents=True, exist_ok=True)


# ---------------- Image transforms ----------------
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])

eval_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


# ---------------- Load datasets ----------------
train_augmented = datasets.ImageFolder(
    TRAIN_DIR, transform=train_transform
)

train_evaluation = datasets.ImageFolder(
    TRAIN_DIR, transform=eval_transform
)

test_dataset = datasets.ImageFolder(
    TEST_DIR, transform=eval_transform
)

class_names = train_augmented.classes

if train_augmented.class_to_idx != train_evaluation.class_to_idx:
    raise ValueError("Training dataset class mappings do not match.")

if train_augmented.class_to_idx != test_dataset.class_to_idx:
    raise ValueError(
        "Training and Testing class mappings differ. "
        "Check the class folders before proceeding."
    )

print("Classes:", class_names)
print("Training images:", len(train_augmented))
print("Testing images:", len(test_dataset))


# ---------------- Stratified train/validation split ----------------
all_indices = list(range(len(train_augmented)))
all_labels = train_augmented.targets

train_indices, val_indices = train_test_split(
    all_indices,
    test_size=0.20,
    random_state=SEED,
    stratify=all_labels,
)

print("Training subset:", len(train_indices))
print("Validation subset:", len(val_indices))

train_subset = Subset(train_augmented, train_indices)
val_subset = Subset(train_evaluation, val_indices)

generator = torch.Generator().manual_seed(SEED)

train_loader = DataLoader(
    train_subset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
    generator=generator,
)

val_loader = DataLoader(
    val_subset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
)


# ---------------- Build model ----------------
model = models.resnet18(weights=ResNet18_Weights.DEFAULT)

# Freeze pretrained feature extractor
for parameter in model.parameters():
    parameter.requires_grad = False

model.fc = nn.Linear(model.fc.in_features, len(class_names))
model = model.to(DEVICE)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(
    model.fc.parameters(),
    lr=LEARNING_RATE,
)


# ---------------- Training and validation ----------------
best_val_accuracy = -1.0

for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    train_correct = 0
    train_total = 0

    for images, labels in train_loader:
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        train_correct += (outputs.argmax(1) == labels).sum().item()
        train_total += labels.size(0)

    train_loss = running_loss / train_total
    train_accuracy = train_correct / train_total

    model.eval()
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)
            val_correct += (outputs.argmax(1) == labels).sum().item()
            val_total += labels.size(0)

    val_accuracy = val_correct / val_total

    print(
        f"Epoch {epoch + 1}/{EPOCHS} | "
        f"Loss: {train_loss:.4f} | "
        f"Train accuracy: {train_accuracy:.4f} | "
        f"Validation accuracy: {val_accuracy:.4f}"
    )

    # Save the checkpoint selected by validation performance
    if val_accuracy > best_val_accuracy:
        best_val_accuracy = val_accuracy
        torch.save(model.state_dict(), MODEL_PATH)

print(f"\nBest validation accuracy: {best_val_accuracy:.4f}")
print(f"Best model saved to: {MODEL_PATH}")

with open(CLASSES_PATH, "w", encoding="utf-8") as file:
    json.dump(class_names, file, indent=2)


# ---------------- Final test evaluation ----------------
# Run this section only after model selection is complete.
model.load_state_dict(
    torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True)
)
model.eval()

all_predictions = []
all_labels = []

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(DEVICE)
        outputs = model(images)
        predictions = outputs.argmax(1).cpu().tolist()

        all_predictions.extend(predictions)
        all_labels.extend(labels.tolist())

test_accuracy = sum(
    p == y for p, y in zip(all_predictions, all_labels)
) / len(all_labels)

print(f"\nFinal test accuracy: {test_accuracy:.4f}")
print("\nClassification report:")
print(
    classification_report(
        all_labels,
        all_predictions,
        labels=list(range(len(class_names))),
        target_names=class_names,
        digits=4,
        zero_division=0,
    )
)

print("Confusion matrix:")
print(confusion_matrix(
    all_labels,
    all_predictions,
    labels=list(range(len(class_names))),
))

print("\nEducational research prototype only—not for medical diagnosis.")
