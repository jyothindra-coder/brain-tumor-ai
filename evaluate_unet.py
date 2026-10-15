
from pathlib import Path
import numpy as np
import nibabel as nib
import torch
import torch.nn.functional as F

from train_unet import UNet, IMAGE_SIZE

DATA_DIR = Path(r"G:\BraTS2020_Sample")
CHECKPOINT_PATH = Path("models/brats_unet_sample.pth")
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

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

# Original BraTS labels are 0, 1, 2, 4.
# Model labels are 0, 1, 2, 3.
original_labels = [0, 1, 2, 4]

case_dirs = [
    DATA_DIR / "BraTS20_Training_009",
    DATA_DIR / "BraTS20_Training_010",
]

# Accumulate voxel counts across complete 3D volumes.
intersection = np.zeros(5, dtype=np.int64)
pred_count = np.zeros(5, dtype=np.int64)
target_count = np.zeros(5, dtype=np.int64)

# Also track per-case whole-tumor Dice.
case_results = []

for case_dir in case_dirs:
    case_id = case_dir.name

    if not case_dir.exists():
        raise FileNotFoundError(f"Missing case: {case_dir}")

    volumes = []

    for modality in checkpoint["modalities"]:
        path = case_dir / f"{case_id}_{modality}.nii"
        volume = np.asarray(
            nib.load(str(path)).dataobj,
            dtype=np.float32,
        )

        # Same normalization as training.
        brain = volume != 0
        if np.any(brain):
            mean = volume[brain].mean()
            std = volume[brain].std()
            volume = (volume - mean) / max(float(std), 1e-6)

        volumes.append(volume)

    seg_path = case_dir / f"{case_id}_seg.nii"
    ground_truth = np.asarray(
        nib.load(str(seg_path)).dataobj,
        dtype=np.uint8,
    )

    case_intersection = 0
    case_pred_count = 0
    case_target_count = 0

    # Predict every axial slice, including slices with no tumor.
    for start in range(0, ground_truth.shape[2], BATCH_SIZE):
        end = min(start + BATCH_SIZE, ground_truth.shape[2])
        batch_slices = []

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

            batch_slices.append(torch.stack(channels))

        x = torch.stack(batch_slices).to(DEVICE)

        with torch.no_grad():
            pred = model(x).argmax(dim=1).cpu().numpy()

        for offset, z in enumerate(range(start, end)):
            target = ground_truth[:, :, z].copy()

            # Map original BraTS label 4 to model label 3.
            target[target == 4] = 3

            target_tensor = F.interpolate(
                torch.from_numpy(
                    target.astype(np.float32)
                )[None, None],
                size=(IMAGE_SIZE, IMAGE_SIZE),
                mode="nearest",
            )[0, 0].numpy().astype(np.uint8)

            pred_slice = pred[offset]

            for class_id in range(1, 4):
                p = pred_slice == class_id
                t = target_tensor == class_id

                intersection[class_id] += np.count_nonzero(p & t)
                pred_count[class_id] += np.count_nonzero(p)
                target_count[class_id] += np.count_nonzero(t)

            # Whole tumor: union of all non-background labels.
            p_whole = pred_slice > 0
            t_whole = target_tensor > 0

            case_intersection += np.count_nonzero(
                p_whole & t_whole
            )
            case_pred_count += np.count_nonzero(p_whole)
            case_target_count += np.count_nonzero(t_whole)

    whole_denom = case_pred_count + case_target_count
    whole_dice = (
        2 * case_intersection / whole_denom
        if whole_denom else 1.0
    )

    case_results.append((case_id, whole_dice))
    print(f"\n{case_id} whole-tumor Dice: {whole_dice:.4f}")

print("\n=== Aggregate Dice scores ===")

scores = []

for class_id, original_label in zip(range(1, 4), [1, 2, 4]):
    denom = pred_count[class_id] + target_count[class_id]

    if target_count[class_id] == 0:
        print(
            f"Label {original_label}: not present in reference; "
            "score not reported"
        )
        continue

    dice = (
        2 * intersection[class_id] / denom
        if denom else 0.0
    )

    scores.append(dice)

    print(
        f"Label {original_label}: Dice={dice:.4f} | "
        f"Reference voxels={target_count[class_id]} | "
        f"Predicted voxels={pred_count[class_id]}"
    )

whole_denom = int(pred_count[1:4].sum() + target_count[1:4].sum())
whole_intersection = int(intersection[1:4].sum())

whole_dice = (
    2 * whole_intersection / whole_denom
    if whole_denom else 1.0
)

print(f"Aggregate whole-tumor Dice: {whole_dice:.4f}")

report_path = OUTPUT_DIR / "unet_evaluation.txt"

with report_path.open("w", encoding="utf-8") as f:
    f.write("BraTS U-Net sample evaluation\n")
    f.write("Patients: 009, 010\n")
    for case_id, score in case_results:
        f.write(f"{case_id} whole-tumor Dice: {score:.4f}\n")
    f.write(f"Aggregate whole-tumor Dice: {whole_dice:.4f}\n")
    for score, label in zip(scores, [1, 2, 4]):
        f.write(f"Label {label} Dice: {score:.4f}\n")

print(f"\nReport saved to: {report_path}")
print("Research prototype only; not for clinical use.")
