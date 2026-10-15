
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from train_unet import UNet

DATA_DIR = Path(r"G:\BraTS2020_Sample")
OUTPUT_DIR = Path("outputs") / "unet_v1_v2_failure_visuals"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PATIENTS = ("003", "005", "008")
MODALITIES = ("flair", "t1", "t1ce", "t2")
DEVICE = "cpu"


def load_model(path):
    checkpoint = torch.load(
        path, map_location=DEVICE, weights_only=True
    )
    model = UNet(num_classes=4)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint


v1, ckpt1 = load_model("models/brats_unet_sample.pth")
v2, ckpt2 = load_model("models/brats_unet_v2.pth")

if ckpt1["image_size"] != ckpt2["image_size"]:
    raise ValueError("The two checkpoints use different image sizes.")

SIZE = int(ckpt1["image_size"])


def normalize(data):
    """Z-score nonzero MRI voxels and leave background at zero."""
    data = data.astype(np.float32, copy=True)
    brain = data != 0

    if np.any(brain):
        mean = data[brain].mean()
        std = data[brain].std()
        data[brain] = (data[brain] - mean) / max(float(std), 1e-6)

    return data


def resize_image(image):
    tensor = torch.from_numpy(
        image.astype(np.float32).copy()
    )[None, None]

    return F.interpolate(
        tensor,
        size=(SIZE, SIZE),
        mode="bilinear",
        align_corners=False,
    )[0, 0]


def resize_mask(mask):
    tensor = torch.from_numpy(
        mask.astype(np.float32).copy()
    )[None, None]

    return F.interpolate(
        tensor,
        size=(SIZE, SIZE),
        mode="nearest",
    )[0, 0].numpy().astype(np.uint8)


def predict_slice(model, volumes, z, normalization):
    channels = []

    for volume in volumes:
        image = volume[:, :, z]

        # v1 was trained with slice-level normalization.
        # v2 was trained with volume-level normalization.
        if normalization == "slice":
            image = normalize(image)

        channels.append(resize_image(image))

    x = torch.stack(channels)[None]

    with torch.no_grad():
        prediction = model(x).argmax(dim=1)[0].numpy()

    # Convert internal label 3 back to BraTS label 4.
    return np.choose(prediction, [0, 1, 2, 4]).astype(np.uint8)


for number in PATIENTS:
    case_id = f"BraTS20_Training_{number}"
    folder = DATA_DIR / case_id

    if not folder.is_dir():
        raise FileNotFoundError(f"Patient folder not found: {folder}")

    raw_volumes = []

    for modality in MODALITIES:
        path = folder / f"{case_id}_{modality}.nii"

        if not path.exists():
            raise FileNotFoundError(f"MRI modality not found: {path}")

        raw_volumes.append(
            np.asarray(nib.load(str(path)).dataobj, dtype=np.float32)
        )

    seg_path = folder / f"{case_id}_seg.nii"

    if not seg_path.exists():
        raise FileNotFoundError(f"Segmentation not found: {seg_path}")

    seg = np.asarray(nib.load(str(seg_path)).dataobj, dtype=np.uint8)

    # v2 uses volume-level normalization.
    volume_normalized = [normalize(v) for v in raw_volumes]

    # Select the slice with the largest reference tumor area.
    areas = np.count_nonzero(seg > 0, axis=(0, 1))
    z = int(np.argmax(areas))

    pred1 = predict_slice(v1, raw_volumes, z, "slice")
    pred2 = predict_slice(v2, volume_normalized, z, "volume")

    target = resize_mask(seg[:, :, z])
    flair = resize_image(volume_normalized[0][:, :, z]).numpy()

    fig, axes = plt.subplots(1, 4, figsize=(19, 5))

    axes[0].imshow(flair.T, cmap="gray", origin="lower")
    axes[0].set_title("FLAIR MRI")

    panels = [
        (axes[1], target, "Reference mask"),
        (axes[2], pred1, "U-Net v1"),
        (axes[3], pred2, "U-Net v2"),
    ]

    for ax, mask, title in panels:
        ax.imshow(flair.T, cmap="gray", origin="lower")
        overlay = np.ma.masked_where(mask.T == 0, mask.T)
        ax.imshow(
            overlay,
            cmap="jet",
            vmin=1,
            vmax=4,
            alpha=0.55,
            origin="lower",
            interpolation="nearest",
        )
        ax.set_title(title)

    for ax in axes:
        ax.axis("off")

    fig.suptitle(f"{case_id} | slice {z}")
    fig.tight_layout()

    output = OUTPUT_DIR / f"{case_id}_v1_v2_comparison.png"
    fig.savefig(output, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved: {output} | slice={z}")

print("All failure-case visualizations completed.")
