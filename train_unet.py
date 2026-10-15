
from pathlib import Path
import random

import nibabel as nib
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

DATA_DIR = Path(r"G:\BraTS2020_Sample")
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

DEVICE = torch.device("cpu")
SEED = 42
BATCH_SIZE = 2
EPOCHS = 3
IMAGE_SIZE = 128

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


class BraTSSliceDataset(Dataset):
    def __init__(self, case_dirs):
        self.samples = []

        for case_dir in case_dirs:
            case_id = case_dir.name
            seg_path = case_dir / f"{case_id}_seg.nii"

            if not seg_path.exists():
                continue

            seg = np.asarray(nib.load(str(seg_path)).dataobj)
            tumor_slices = np.where(np.any(seg > 0, axis=(0, 1)))[0]

            # Limit initial training to tumor-containing slices.
            for z in tumor_slices:
                self.samples.append((case_dir, int(z)))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        case_dir, z = self.samples[index]
        case_id = case_dir.name
        channels = []

        for modality in ("flair", "t1", "t1ce", "t2"):
            path = case_dir / f"{case_id}_{modality}.nii"
            volume = np.asarray(nib.load(str(path)).dataobj, dtype=np.float32)
            image = volume[:, :, z]

            # Normalize non-zero brain voxels.
            brain = image != 0
            if np.any(brain):
                mean = image[brain].mean()
                std = image[brain].std()
                image = (image - mean) / max(float(std), 1e-6)

            # Resize to reduce CPU memory and computation.
            image = torch.from_numpy(image.copy())[None, None]
            image = torch.nn.functional.interpolate(
                image, size=(IMAGE_SIZE, IMAGE_SIZE),
                mode="bilinear", align_corners=False
            )[0, 0]

            channels.append(image)

        seg_path = case_dir / f"{case_id}_seg.nii"
        seg = np.asarray(nib.load(str(seg_path)).dataobj)
        mask = seg[:, :, z].astype(np.uint8)

        # Original BraTS labels: 0, 1, 2, 4.
        # Training labels: 0, 1, 2, 3.
        mask[mask == 4] = 3
        mask = torch.from_numpy(mask.astype(np.float32))[None, None]
        mask = torch.nn.functional.interpolate(
            mask, size=(IMAGE_SIZE, IMAGE_SIZE), mode="nearest"
        )[0, 0].long()

        x = torch.stack(channels)
        return x, mask


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.layers(x)


class UNet(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()
        self.enc1 = DoubleConv(4, 16)
        self.enc2 = DoubleConv(16, 32)
        self.enc3 = DoubleConv(32, 64)
        self.pool = nn.MaxPool2d(2)
        self.up2 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.dec2 = DoubleConv(64, 32)
        self.up1 = nn.ConvTranspose2d(32, 16, 2, stride=2)
        self.dec1 = DoubleConv(32, 16)
        self.out = nn.Conv2d(16, num_classes, 1)

    def forward(self, x):
        a = self.enc1(x)
        b = self.enc2(self.pool(a))
        c = self.enc3(self.pool(b))
        x = self.dec2(torch.cat([self.up2(c), b], dim=1))
        x = self.dec1(torch.cat([self.up1(x), a], dim=1))
        return self.out(x)


def main():
    cases = sorted(
        p for p in DATA_DIR.iterdir()
        if p.is_dir() and (p / f"{p.name}_seg.nii").exists()
    )

    if len(cases) < 3:
        raise RuntimeError(
            f"Expected at least 3 BraTS cases; found {len(cases)}"
        )

    # Split by patient, never by slice.
    random.shuffle(cases)
    split = max(1, int(len(cases) * 0.8))
    train_cases = cases[:split]
    val_cases = cases[split:]

    train_data = BraTSSliceDataset(train_cases)
    val_data = BraTSSliceDataset(val_cases)

    if len(train_data) == 0 or len(val_data) == 0:
        raise RuntimeError("No tumor-containing slices found in a split.")

    print("Device:", DEVICE)
    print("Training patients:", len(train_cases))
    print("Validation patients:", len(val_cases))
    print("Training slices:", len(train_data))
    print("Validation slices:", len(val_data))

    train_loader = DataLoader(
        train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        val_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=0
    )

    model = UNet().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    best_dice = -1.0

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0

        for images, masks in train_loader:
            images, masks = images.to(DEVICE), masks.to(DEVICE)
            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, masks)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        intersection = torch.zeros(4)
        denominator = torch.zeros(4)

        with torch.no_grad():
            for images, masks in val_loader:
                logits = model(images.to(DEVICE))
                predictions = logits.argmax(dim=1)

                for class_id in range(1, 4):
                    pred = predictions == class_id
                    target = masks == class_id
                    intersection[class_id] += (pred & target).sum().item()
                    denominator[class_id] += pred.sum().item() + target.sum().item()

        dice_scores = []
        for class_id in range(1, 4):
            if denominator[class_id] > 0:
                dice_scores.append(
                    (2 * intersection[class_id] / denominator[class_id]).item()
                )

        mean_dice = float(np.mean(dice_scores)) if dice_scores else 0.0
        average_loss = train_loss / len(train_loader)

        print(
            f"Epoch {epoch + 1}/{EPOCHS} | "
            f"Train loss: {average_loss:.4f} | "
            f"Validation Dice: {mean_dice:.4f}"
        )

        if mean_dice > best_dice:
            best_dice = mean_dice
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "num_classes": 4,
                    "image_size": IMAGE_SIZE,
                    "original_labels": [0, 1, 2, 4],
                    "modalities": ["flair", "t1", "t1ce", "t2"],
                },
                MODEL_DIR / "brats_unet_sample.pth",
            )

    print("Best validation Dice:", round(best_dice, 4))
    print("Saved model: models/brats_unet_sample.pth")
    print("Research prototype only; not for clinical use.")


if __name__ == "__main__":
    main()
