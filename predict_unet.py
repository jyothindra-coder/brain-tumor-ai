
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
import matplotlib.pyplot as plt

from train_unet import UNet, IMAGE_SIZE

CASE_DIR = Path(
    r"G:\BraTS2020_Sample\BraTS20_Training_009"
)
CASE_ID = CASE_DIR.name
CHECKPOINT = Path("models/brats_unet_sample.pth")
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

device = torch.device("cpu")

checkpoint = torch.load(
    CHECKPOINT, map_location=device, weights_only=True
)

model = UNet(num_classes=checkpoint["num_classes"])
model.load_state_dict(checkpoint["model_state_dict"])
model.to(device)
model.eval()

modalities = checkpoint["modalities"]
channels = []
volumes = {}

for modality in modalities:
    path = CASE_DIR / f"{CASE_ID}_{modality}.nii"
    volume = np.asarray(
        nib.load(str(path)).dataobj, dtype=np.float32
    )
    volumes[modality] = volume

    image = volume[:, :, :]
    brain = image != 0

    if np.any(brain):
        mean = image[brain].mean()
        std = image[brain].std()
        image = (image - mean) / max(float(std), 1e-6)

    channels.append(image)

seg_path = CASE_DIR / f"{CASE_ID}_seg.nii"
ground_truth = np.asarray(
    nib.load(str(seg_path)).dataobj
)

# Select a slice containing the largest reference tumor area.
tumor_area = np.count_nonzero(
    ground_truth > 0, axis=(0, 1)
)
z = int(np.argmax(tumor_area))

input_slices = []

for image in channels:
    tensor = torch.from_numpy(
        image[:, :, z].copy()
    )[None, None]

    tensor = torch.nn.functional.interpolate(
        tensor,
        size=(IMAGE_SIZE, IMAGE_SIZE),
        mode="bilinear",
        align_corners=False,
    )
    input_slices.append(tensor[0, 0])

x = torch.stack(input_slices)[None].to(device)

with torch.no_grad():
    prediction = model(x).argmax(dim=1)[0].cpu().numpy()

# Restore original BraTS labels for display.
prediction_original = np.choose(
    prediction, [0, 1, 2, 4]
)

# Nearest-neighbor resizing preserves discrete mask labels.
gt_slice = ground_truth[:, :, z].astype(np.uint8)
gt_resized = torch.nn.functional.interpolate(
    torch.from_numpy(gt_slice.astype(np.float32))[None, None],
    size=(IMAGE_SIZE, IMAGE_SIZE),
    mode="nearest",
)[0, 0].numpy().astype(np.uint8)

flair = volumes["flair"][:, :, z]
flair_tensor = torch.from_numpy(
    flair.copy()
)[None, None]

flair_resized = torch.nn.functional.interpolate(
    flair_tensor,
    size=(IMAGE_SIZE, IMAGE_SIZE),
    mode="bilinear",
    align_corners=False,
)[0, 0].numpy()

plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.imshow(flair_resized.T, cmap="gray", origin="lower")
plt.title(f"FLAIR MRI — Slice {z}")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(flair_resized.T, cmap="gray", origin="lower")
plt.imshow(
    np.ma.masked_where(gt_resized.T == 0, gt_resized.T),
    cmap="jet", alpha=0.55, vmin=1, vmax=4,
    origin="lower", interpolation="nearest"
)
plt.title("Reference Mask")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(flair_resized.T, cmap="gray", origin="lower")
plt.imshow(
    np.ma.masked_where(
        prediction_original.T == 0,
        prediction_original.T
    ),
    cmap="jet", alpha=0.55, vmin=1, vmax=4,
    origin="lower", interpolation="nearest"
)
plt.title("U-Net Prediction")
plt.axis("off")

plt.tight_layout()
out_path = OUTPUT_DIR / "brats_unet_prediction.png"
plt.savefig(out_path, dpi=150)
plt.show()

print("Case:", CASE_ID)
print("Slice:", z)
print("Saved visualization:", out_path)
print("Predicted labels:", np.unique(prediction_original))
