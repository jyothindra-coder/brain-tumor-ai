
import json
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from torchvision.models import ResNet18_Weights
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import accuracy_score

# 1. Configuration
DATA_DIR = Path("data")
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

BATCH_SIZE = 8
EPOCHS = 3
LEARNING_RATE = 0.001

device = torch.device("cpu")
print("Using device:", device)

# 2. Image preprocessing
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# 3. Load images from folders
train_data = datasets.ImageFolder(
    DATA_DIR / "Training",
    transform=train_transform
)

test_data = datasets.ImageFolder(
    DATA_DIR / "Testing",
    transform=test_transform
)

if train_data.classes != test_data.classes:
    raise ValueError("Training and testing classes do not match.")

class_names = train_data.classes
num_classes = len(class_names)

print("Classes:", class_names)
print("Training images:", len(train_data))
print("Testing images:", len(test_data))

train_loader = DataLoader(
    train_data,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)

test_loader = DataLoader(
    test_data,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

# 4. Load pretrained ResNet18
weights = ResNet18_Weights.DEFAULT
model = models.resnet18(weights=weights)

# Train only the final classification layer initially
for parameter in model.parameters():
    parameter.requires_grad = False

model.fc = nn.Linear(
    model.fc.in_features,
    num_classes
)

model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(
    model.fc.parameters(),
    lr=LEARNING_RATE
)

# 5. Train the model
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        predictions = outputs.argmax(dim=1)
        correct += (predictions == labels).sum().item()
        total += labels.size(0)

    print(
        f"Epoch {epoch + 1}/{EPOCHS} | "
        f"Loss: {total_loss / len(train_loader):.4f} | "
        f"Training accuracy: {correct / total:.4f}"
    )

# 6. Evaluate on the provided test folder
model.eval()
all_labels = []
all_predictions = []

with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images.to(device))
        predictions = outputs.argmax(dim=1)

        all_labels.extend(labels.tolist())
        all_predictions.extend(predictions.tolist())

print("\nTest accuracy:",
      round(accuracy_score(all_labels, all_predictions), 4))

print("\nClassification report:")
print(classification_report(
    all_labels,
    all_predictions,
    labels=list(range(num_classes)),
    target_names=class_names,
    zero_division=0
))

print("Confusion matrix:")
print(confusion_matrix(
    all_labels,
    all_predictions,
    labels=list(range(num_classes))
))

# 7. Save the model and its class labels
torch.save(
    model.state_dict(),
    MODEL_DIR / "brain_tumor_resnet18.pth"
)

with open(MODEL_DIR / "classes.json", "w") as file:
    json.dump(class_names, file)

print("\nSaved model to models/brain_tumor_resnet18.pth")