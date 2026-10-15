
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F

from train_unet import UNet

DATA_DIR = Path(r"G:\BraTS2020_Sample")
CHECKPOINT_PATH = Path("models/brats_unet_v2.pth")
OUTPUT_PATH = Path("outputs/unet_v2_evaluation.txt")
IMAGE_SIZE = 128
DEVICE = torch.device("cpu")
BATCH_SIZE = 4

checkpoint = torch.load(
    CHECKPOINT_PATH,
    map_location=DEVICE,
    weights_only=True,
)

model = UNet(num_classes=checkpoint["num_classes"])
model.load_state_dict(checkpoint["model_state_dict"])
model.to(DEVICE)
model.eval()

modalities = checkpoint["modalities"]

# Model labels: 0, 1, 2, 3.
# Original BraTS labels: 0, 1, 2, 4.
label_map = {1: 1, 2: 2, 3: 4}

# Store voxel counts across both held-out patients.
intersection = {k: 0 for k in label_map}
pred_count = {k: 0 for k in label_map}
target_count = {k: 0 for k in label_map}

case_results = []

for case_number in ("001", "002"):
    case_id = f"BraTS20_Training_{case_number}"
    case_dir = DATA_DIR / case_id

    if not case_dir.exists():
        raise FileNotFoundError(f"Missing case folder: {case_dir}")

    volumes = []

    for modality in modalities:
        path = case_dir / f"{case_id}_{modality}.nii"
        volume = np.asarray(
            nib.load(str(path)).dataobj,
            dtype=np.float32,
        )

        # Match training normalization.
        brain = volume != 0
        if np.any(brain):
            mean = volume[brain].mean()
            std = volume[brain].std()
            volume = (volume - mean) / max(float(std), 1e-6)

        volumes.append(volume)

    seg_path = case_dir / f"{case_id}_seg.nii"
    reference = np.asarray(
        nib.load(str(seg_path)).dataobj,
        dtype=np.uint8,
    )

    case_intersection = 0
    case_pred_count = 0
    case_target_count = 0

    print(f"Evaluating {case_id}...")

    for start in range(0, reference.shape[2], BATCH_SIZE):
        end = min(start + BATCH_SIZE, reference.shape[2])
        batch = []

        for z in range(start, end):
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

            batch.append(torch.stack(channels))

        x = torch.stack(batch).to(DEVICE)

        with torch.no_grad():
            predictions = model(x).argmax(dim=1).cpu().numpy()

        for offset, z in enumerate(range(start, end)):
            target = reference[:, :, z].copy()

            # Resize the reference mask using nearest neighbor.
            target_tensor = torch.from_numpy(
                target.astype(np.float32)
            )[None, None]

            target_small = F.interpolate(
                target_tensor,
                size=(IMAGE_SIZE, IMAGE_SIZE),
                mode="nearest",
            )[0, 0].numpy().astype(np.uint8)

            pred = predictions[offset]

            for model_label, original_label in label_map.items():
                p = pred == model_label
                t = target_small == original_label

                intersection[model_label] += int(np.count_nonzero(p & t))
                pred_count[model_label] += int(np.count_nonzero(p))
                target_count[model_label] += int(np.count_nonzero(t))

            p_whole = pred > 0
            t_whole = target_small > 0

            case_intersection += int(np.count_nonzero(p_whole & t_whole))
            case_pred_count += int(np.count_nonzero(p_whole))
            case_target_count += int(np.count_nonzero(t_whole))

    denom = case_pred_count + case_target_count
    whole_dice = 2 * case_intersection / denom if denom else 1.0
    case_results.append((case_id, whole_dice))

    print(f"{case_id} whole-tumor Dice: {whole_dice:.4f}")

print("\n=== Aggregate results across held-out patients ===")

lines = [
    "BraTS U-Net v2 evaluation",
    "Held-out patients: 001, 002",
    "Resolution: 128 x 128",
]

for case_id, score in case_results:
    lines.append(f"{case_id} whole-tumor Dice: {score:.4f}")

scores = []

for model_label, original_label in label_map.items():
    denom = pred_count[model_label] + target_count[model_label]

    if target_count[model_label] == 0:
        message = (
            f"Label {original_label}: absent from reference; "
            "Dice not reported"
        )
    else:
        score = (
            2 * intersection[model_label] / denom
            if denom else 0.0
        )
        scores.append(score)
        message = (
            f"Label {original_label}: Dice={score:.4f} | "
            f"Reference voxels={target_count[model_label]} | "
            f"Predicted voxels={pred_count[model_label]}"
        )

    print(message)
    lines.append(message)

whole_intersection = sum(intersection.values())
whole_pred = sum(pred_count.values())
whole_target = sum(target_count.values())
whole_denom = whole_pred + whole_target

whole_dice = (
    2 * whole_intersection / whole_denom
    if whole_denom else 1.0
)

print(f"Aggregate whole-tumor Dice: {whole_dice:.4f}")
lines.append(f"Aggregate whole-tumor Dice: {whole_dice:.4f}")

if scores:
    macro_dice = float(np.mean(scores))
    print(f"Aggregate tumor-label macro Dice: {macro_dice:.4f}")
    lines.append(f"Aggregate tumor-label macro Dice: {macro_dice:.4f}")

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
OUTPUT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")

print(f"\nReport saved to: {OUTPUT_PATH}")
print("Research prototype only; not for clinical use.")
