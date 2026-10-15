
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from train_unet import UNet

DATA_DIR = Path(r"G:\BraTS2020_Sample")
OUTPUT_DIR = Path("outputs") / "unet_v2_visuals"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

checkpoint = torch.load(
    "models/brats_unet_v2.pth",
    map_location="cpu",
    weights_only=True,
)

model = UNet(num_classes=4)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

size = checkpoint["image_size"]
modalities = checkpoint["modalities"]

for number in ("001", "002"):
    case_id = f"BraTS20_Training_{number}"
    folder = DATA_DIR / case_id

    volumes = []
    for modality in modalities:
        path = folder / f"{case_id}_{modality}.nii"
        volume = np.asarray(
            nib.load(str(path)).dataobj,
            dtype=np.float32,
        )

        brain = volume != 0
        if np.any(brain):
            mean = volume[brain].mean()
            std = volume[brain].std()
            volume = (volume - mean) / max(float(std), 1e-6)

        volumes.append(volume)

    seg = np.asarray(
        nib.load(str(folder / f"{case_id}_seg.nii")).dataobj
    )

    # Choose the slice with the largest reference tumor area.
    areas = np.count_nonzero(seg > 0, axis=(0, 1))
    z = int(np.argmax(areas))

    channels = []
    for volume in volumes:
        tensor = torch.from_numpy(
            volume[:, :, z].copy()
        )[None, None]

        tensor = F.interpolate(
            tensor,
            size=(size, size),
            mode="bilinear",
            align_corners=False,
        )[0, 0]
        channels.append(tensor)

    x = torch.stack(channels)[None]

    with torch.no_grad():
        pred = model(x).argmax(dim=1)[0].numpy()

    # Convert model label 3 back to BraTS label 4.
    pred_original = np.choose(pred, [0, 1, 2, 4])

    target = seg[:, :, z].astype(np.uint8)
    target_small = F.interpolate(
        torch.from_numpy(target.astype(np.float32))[None, None],
        size=(size, size),
        mode="nearest",
    )[0, 0].numpy().astype(np.uint8)

    flair = torch.from_numpy(
        volumes[0][:, :, z].copy()
    )[None, None]
    flair = F.interpolate(
        flair, size=(size, size),
        mode="bilinear", align_corners=False,
    )[0, 0].numpy()

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(flair.T, cmap="gray", origin="lower")
    axes[0].set_title(f"{case_id}: FLAIR")

    for ax, mask, title in [
        (axes[1], target_small, "Reference mask"),
        (axes[2], pred_original, "U-Net v2 prediction"),
    ]:
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

    fig.tight_layout()
    output = OUTPUT_DIR / f"{case_id}_comparison.png"
    fig.savefig(output, dpi=150)
    plt.close(fig)

    print(f"Saved {output} | slice={z}")

print("Visualization complete.")
