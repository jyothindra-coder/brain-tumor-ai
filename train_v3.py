
from pathlib import Path
import json
import random

import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, models, transforms
from torchvision.models import ResNet18_Weights
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)


# --------------------------------------------------
# Configuration
# --------------------------------------------------

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"

TRAIN_DIR = DATA_DIR / "Training"
TEST_DIR = DATA_DIR / "Testing"

MODEL_PATH = MODEL_DIR / "brain_tumor_resnet18_v3.pth"
CLASSES_PATH = MODEL_DIR / "classes_v3.json"

SEED = 42
BATCH_SIZE = 8
MAX_EPOCHS = 8
PATIENCE = 3

# Smaller updates for pretrained features;
# larger updates for the new classification layer.
LAYER4_LR = 0.00001
CLASSIFIER_LR = 0.0001

DEVICE = torch.device("cpu")

random.seed(SEED)
torch.manual_seed(SEED)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

print("Device:", DEVICE)
print("Training a new model; existing models will not be changed.")


# --------------------------------------------------
# Image transformations
# --------------------------------------------------

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
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


# --------------------------------------------------
# Load datasets
# --------------------------------------------------

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
num_classes = len(class_names)

if train_augmented.class_to_idx != train_evaluation.class_to_idx:
    raise ValueError("Training class mappings do not match.")

if train_augmented.class_to_idx != test_dataset.class_to_idx:
    raise ValueError("Training and testing classes do not match.")

if len(train_augmented) == 0 or len(test_dataset) == 0:
    raise ValueError("Training or testing dataset is empty.")

print("Classes:", class_names)
print("Training images:", len(train_augmented))
print("Test images:", len(test_dataset))


# --------------------------------------------------
# Stratified train/validation split
# --------------------------------------------------

indices = list(range(len(train_augmented)))
labels = train_augmented.targets

train_indices, val_indices = train_test_split(
    indices,
    test_size=0.20,
    random_state=SEED,
    stratify=labels,
)

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

print("Training subset:", len(train_subset))
print("Validation subset:", len(val_subset))


# --------------------------------------------------
# Build pretrained ResNet18
# --------------------------------------------------

model = models.resnet18(weights=ResNet18_Weights.DEFAULT)

# Freeze all pretrained parameters first.
for parameter in model.parameters():
    parameter.requires_grad = False

# Fine-tune the deeper feature block.
for parameter in model.layer4.parameters():
    parameter.requires_grad = True

# Create a new output layer for our four classes.
model.fc = nn.Linear(model.fc.in_features, num_classes)

model = model.to(DEVICE)

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam([
    {
        "params": model.layer4.parameters(),
        "lr": LAYER4_LR,
    },
    {
        "params": model.fc.parameters(),
        "lr": CLASSIFIER_LR,
    },
])


# --------------------------------------------------
# Training and validation
# --------------------------------------------------

best_val_f1 = -1.0
best_val_loss = float("inf")
epochs_without_improvement = 0

for epoch in range(MAX_EPOCHS):
    model.train()

    # Keep frozen feature blocks and their BatchNorm statistics fixed.
    model.conv1.eval()
    model.bn1.eval()
    model.layer1.eval()
    model.layer2.eval()
    model.layer3.eval()

    running_loss = 0.0
    train_correct = 0
    train_total = 0

    for images, targets in train_loader:
        images = images.to(DEVICE)
        targets = targets.to(DEVICE)

        optimizer.zero_grad(set_to_none=True)

        outputs = model(images)
        loss = criterion(outputs, targets)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        train_correct += (
            outputs.argmax(dim=1) == targets
        ).sum().item()
        train_total += targets.size(0)

    train_loss = running_loss / train_total
    train_accuracy = train_correct / train_total

    # Evaluate on validation data only for checkpoint selection.
    model.eval()

    val_loss_total = 0.0
    val_actual = []
    val_predicted = []

    with torch.inference_mode():
        for images, targets in val_loader:
            outputs = model(images.to(DEVICE))
            loss = criterion(outputs, targets.to(DEVICE))

            val_loss_total += loss.item() * images.size(0)

            predictions = outputs.argmax(dim=1).cpu().tolist()
            val_predicted.extend(predictions)
            val_actual.extend(targets.tolist())

    val_loss = val_loss_total / len(val_subset)
    val_accuracy = accuracy_score(val_actual, val_predicted)
    val_macro_f1 = f1_score(
        val_actual,
        val_predicted,
        labels=list(range(num_classes)),
        average="macro",
        zero_division=0,
    )

    print(
        f"Epoch {epoch + 1}/{MAX_EPOCHS} | "
        f"Train loss: {train_loss:.4f} | "
        f"Train accuracy: {train_accuracy:.4f} | "
        f"Val loss: {val_loss:.4f} | "
        f"Val accuracy: {val_accuracy:.4f} | "
        f"Val macro-F1: {val_macro_f1:.4f}"
    )

    improved = (
        val_macro_f1 > best_val_f1
        or (
            val_macro_f1 == best_val_f1
            and val_loss < best_val_loss
        )
    )

    if improved:
        best_val_f1 = val_macro_f1
        best_val_loss = val_loss
        epochs_without_improvement = 0

        torch.save(model.state_dict(), MODEL_PATH)

        print("  Saved new best validation checkpoint.")
    else:
        epochs_without_improvement += 1
        print(
            "  No improvement:",
            epochs_without_improvement,
            "of",
            PATIENCE,
        )

    if epochs_without_improvement >= PATIENCE:
        print("Early stopping: validation metric did not improve.")
        break


with open(CLASSES_PATH, "w", encoding="utf-8") as file:
    json.dump(class_names, file, indent=2)

print("\nBest validation macro-F1:", round(best_val_f1, 4))
print("Saved model:", MODEL_PATH)
print("Saved classes:", CLASSES_PATH)


# --------------------------------------------------
# Final test evaluation
# --------------------------------------------------
# This test set has already been used for earlier comparisons.
# Treat these results as a final diagnostic, not as a tuning target.

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=DEVICE,
        weights_only=True,
    )
)
model.eval()

test_actual = []
test_predicted = []

with torch.inference_mode():
    for images, targets in test_loader:
        outputs = model(images.to(DEVICE))
        predictions = outputs.argmax(dim=1).cpu().tolist()

        test_predicted.extend(predictions)
        test_actual.extend(targets.tolist())

test_accuracy = accuracy_score(test_actual, test_predicted)
test_macro_f1 = f1_score(
    test_actual,
    test_predicted,
    labels=list(range(num_classes)),
    average="macro",
    zero_division=0,
)

print("\nFinal test accuracy:", round(test_accuracy, 4))
print("Final test macro-F1:", round(test_macro_f1, 4))

print("\nClassification report:")
print(classification_report(
    test_actual,
    test_predicted,
    labels=list(range(num_classes)),
    target_names=class_names,
    digits=4,
    zero_division=0,
))

print("Confusion matrix (rows=actual, columns=predicted):")
print(confusion_matrix(
    test_actual,
    test_predicted,
    labels=list(range(num_classes)),
))

print(
    "\nEducational research prototype only. "
    "Not for medical diagnosis."
)
