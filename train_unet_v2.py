
from pathlib import Path
import random

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import Dataset, DataLoader

from train_unet import UNet

DATA_DIR = Path(r"G:\BraTS2020_Sample")
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

IMAGE_SIZE = 128
BATCH_SIZE = 2
EPOCHS = 8
PATIENCE = 3
SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = torch.device("cpu")
MODALITIES = ["flair", "t1", "t1ce", "t2"]


class BraTSDataset(Dataset):
    def __init__(self, case_dirs):
        self.images = []
        self.masks = []

        for case_dir in case_dirs:
            case_id = case_dir.name
            print("Preparing:", case_id)

            volumes = []

            for modality in MODALITIES:
                path = case_dir / f"{case_id}_{modality}.nii"
                volume = np.asarray(
                    nib.load(str(path)).dataobj,
                    dtype=np.float32,
                )

                brain = volume != 0
                if np.any(brain):
                    mean = volume[brain].mean()
                    std = volume[brain].std()
                    volume = (
                        volume - mean
                    ) / max(float(std), 1e-6)

                volumes.append(volume)

            seg = np.asarray(
                nib.load(
                    str(case_dir / f"{case_id}_seg.nii")
                ).dataobj,
                dtype=np.uint8,
            )

            # Keep every tumor-containing slice and sample
            # one background slice in approximately every five.
            for z in range(seg.shape[2]):
                has_tumor = np.any(seg[:, :, z] > 0)

                if not has_tumor and z % 5 != 0:
                    continue

                channels = []

                for volume in volumes:
                    image = torch.from_numpy(
                        volume[:, :, z].copy()
                    )[None, None]

                    image = F.interpolate(
                        image,
                        size=(IMAGE_SIZE, IMAGE_SIZE),
                        mode="bilinear",
                        align_corners=False,
                    )[0, 0]

                    channels.append(image)

                mask = seg[:, :, z].copy()
                mask[mask == 4] = 3

                mask = torch.from_numpy(
                    mask.astype(np.float32)
                )[None, None]

                mask = F.interpolate(
                    mask,
                    size=(IMAGE_SIZE, IMAGE_SIZE),
                    mode="nearest",
                )[0, 0].long()

                self.images.append(torch.stack(channels))
                self.masks.append(mask)

        if not self.images:
            raise RuntimeError("No usable slices were found.")

        self.images = torch.stack(self.images)
        self.masks = torch.stack(self.masks)

        print("Prepared slices:", len(self.images))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        return self.images[index], self.masks[index]


def combined_loss(logits, targets):
    # Cross-entropy learns all four pixel classes.
    ce = F.cross_entropy(logits, targets)

    # Soft Dice encourages overlap, including smaller regions.
    probabilities = torch.softmax(logits, dim=1)
    one_hot = F.one_hot(
        targets, num_classes=4
    ).permute(0, 3, 1, 2).float()

    dims = (0, 2, 3)
    intersection = (probabilities * one_hot).sum(dims)
    denominator = probabilities.sum(dims) + one_hot.sum(dims)

    dice = (2 * intersection + 1e-5) / (denominator + 1e-5)

    # Ignore background in the Dice component.
    dice_loss = 1 - dice[1:].mean()

    return ce + dice_loss


def evaluate(model, loader):
    model.eval()

    intersection = torch.zeros(4, dtype=torch.float64)
    denominator = torch.zeros(4, dtype=torch.float64)
    target_count = torch.zeros(4, dtype=torch.float64)

    with torch.no_grad():
        for images, masks in loader:
            predictions = model(images.to(DEVICE)).argmax(dim=1)
            predictions = predictions.cpu()

            for class_id in range(1, 4):
                pred = predictions == class_id
                target = masks == class_id

                intersection[class_id] += (pred & target).sum().item()
                denominator[class_id] += (
                    pred.sum().item() + target.sum().item()
                )
                target_count[class_id] += target.sum().item()

    scores = []

    for class_id in range(1, 4):
        if target_count[class_id] == 0:
            continue

        score = (
            2 * intersection[class_id] / denominator[class_id]
            if denominator[class_id] > 0 else 0.0
        )
        scores.append(float(score))

    return float(np.mean(scores)) if scores else 0.0


def main():
    cases = sorted(
        p for p in DATA_DIR.iterdir()
        if p.is_dir()
        and (p / f"{p.name}_seg.nii").exists()
    )

    if len(cases) < 5:
        raise RuntimeError(
            "At least 5 patient cases are required for this experiment."
        )

    # Fixed patient-level split for repeatable comparisons.
    random.Random(SEED).shuffle(cases)
    split = max(1, int(len(cases) * 0.8))

    train_cases = cases[:split]
    val_cases = cases[split:]

    print("Training patients:", [p.name for p in train_cases])
    print("Validation patients:", [p.name for p in val_cases])

    train_data = BraTSDataset(train_cases)
    val_data = BraTSDataset(val_cases)

    train_loader = DataLoader(
        train_data,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
    )
    val_loader = DataLoader(
        val_data,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    model = UNet(num_classes=4).to(DEVICE)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=0.0005
    )

    best_dice = -1.0
    epochs_without_improvement = 0
    checkpoint_path = MODEL_DIR / "brats_unet_v2.pth"

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0

        for images, masks in train_loader:
            images = images.to(DEVICE)
            masks = masks.to(DEVICE)

            optimizer.zero_grad()
            logits = model(images)
            loss = combined_loss(logits, masks)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        val_dice = evaluate(model, val_loader)
        average_loss = total_loss / len(train_loader)

        print(
            f"Epoch {epoch + 1}/{EPOCHS} | "
            f"Train loss: {average_loss:.4f} | "
            f"Validation macro Dice: {val_dice:.4f}"
        )

        if val_dice > best_dice:
            best_dice = val_dice
            epochs_without_improvement = 0

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "num_classes": 4,
                    "image_size": IMAGE_SIZE,
                    "original_labels": [0, 1, 2, 4],
                    "modalities": MODALITIES,
                    "validation_macro_dice": best_dice,
                    "seed": SEED,
                },
                checkpoint_path,
            )
            print("Saved best checkpoint.")
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= PATIENCE:
            print("Early stopping.")
            break

    print("Best validation macro Dice:", round(best_dice, 4))
    print("Checkpoint:", checkpoint_path)
    print("Research prototype only; not for clinical use.")


if __name__ == "__main__":
    main()
