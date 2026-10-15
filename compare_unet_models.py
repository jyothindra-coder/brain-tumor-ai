
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F

from train_unet import UNet

DATA_DIR = Path(r"G:\BraTS2020_Sample")
OUTPUT_PATH = Path("outputs/unet_all10_comparison.txt")

DEVICE = torch.device("cpu")
IMAGE_SIZE = 128
BATCH_SIZE = 4
MODALITIES = ("flair", "t1", "t1ce", "t2")
PATIENTS = ("001", "002", "003", "004", "005", "006", "007", "008", "009", "010")

# Original BraTS label -> model class.
ORIGINAL_TO_MODEL = {0: 0, 1: 1, 2: 2, 4: 3}
MODEL_TO_ORIGINAL = {v: k for k, v in ORIGINAL_TO_MODEL.items()}

MODELS = {
    "U-Net v1": {
        "path": Path("models/brats_unet_sample.pth"),
        "normalization": "slice",
    },
    "U-Net v2": {
        "path": Path("models/brats_unet_v2.pth"),
        "normalization": "volume",
    },
}


def normalize(image):
    """Normalize non-zero voxels; leave background at zero."""
    result = image.astype(np.float32, copy=True)
    brain = result != 0

    if np.any(brain):
        values = result[brain]
        mean = float(values.mean())
        std = float(values.std())
        result[brain] = (values - mean) / max(std, 1e-6)

    return result


def load_case(case_id):
    folder = DATA_DIR / case_id

    volumes = {}
    for modality in MODALITIES:
        path = folder / f"{case_id}_{modality}.nii"
        if not path.exists():
            raise FileNotFoundError(path)

        volumes[modality] = np.asarray(
            nib.load(str(path)).dataobj,
            dtype=np.float32,
        )

    seg_path = folder / f"{case_id}_seg.nii"
    if not seg_path.exists():
        raise FileNotFoundError(seg_path)

    reference = np.asarray(
        nib.load(str(seg_path)).dataobj,
        dtype=np.uint8,
    )

    unexpected = set(np.unique(reference).tolist()) - {0, 1, 2, 4}
    if unexpected:
        raise ValueError(
            f"Unexpected BraTS labels in {case_id}: {unexpected}"
        )

    return volumes, reference


def prepare_slice(volumes, z, normalization):
    channels = []

    for modality in MODALITIES:
        volume = volumes[modality]

        if normalization == "volume":
            # v2 training normalized each complete MRI volume.
            image = volume[:, :, z]
        elif normalization == "slice":
            # v1 training normalized each individual 2D slice.
            image = volume[:, :, z]
            image = normalize(image)
        else:
            raise ValueError(f"Unknown normalization: {normalization}")

        tensor = torch.from_numpy(
            np.asarray(image, dtype=np.float32).copy()
        )[None, None]

        tensor = F.interpolate(
            tensor,
            size=(IMAGE_SIZE, IMAGE_SIZE),
            mode="bilinear",
            align_corners=False,
        )[0, 0]

        channels.append(tensor)

    return torch.stack(channels)


def resize_reference(reference_slice):
    # Preserve discrete labels with nearest-neighbor interpolation.
    model_labels = reference_slice.copy()
    model_labels[model_labels == 4] = 3

    tensor = torch.from_numpy(
        model_labels.astype(np.float32)
    )[None, None]

    return F.interpolate(
        tensor,
        size=(IMAGE_SIZE, IMAGE_SIZE),
        mode="nearest",
    )[0, 0].numpy().astype(np.uint8)


def evaluate_model(model, volumes, reference, normalization):
    # For v2, normalize each complete volume once, matching training.
    if normalization == "volume":
        volumes = {
            modality: normalize(volume)
            for modality, volume in volumes.items()
        }

    # Counts aggregate over the complete 3D volume after resizing
    # each axial slice to the common 128 x 128 evaluation resolution.
    intersections = np.zeros(4, dtype=np.int64)
    pred_counts = np.zeros(4, dtype=np.int64)
    target_counts = np.zeros(4, dtype=np.int64)

    model.eval()

    for start in range(0, reference.shape[2], BATCH_SIZE):
        end = min(start + BATCH_SIZE, reference.shape[2])

        batch = torch.stack([
            prepare_slice(volumes, z, normalization)
            for z in range(start, end)
        ]).to(DEVICE)

        with torch.no_grad():
            predictions = model(batch).argmax(dim=1).cpu().numpy()

        for offset, z in enumerate(range(start, end)):
            pred = predictions[offset]
            target = resize_reference(reference[:, :, z])

            for class_id in (1, 2, 3):
                p = pred == class_id
                t = target == class_id

                intersections[class_id] += np.count_nonzero(p & t)
                pred_counts[class_id] += np.count_nonzero(p)
                target_counts[class_id] += np.count_nonzero(t)

    # Combine all three tumor labels for whole-tumor metrics.
    whole_intersection = int(intersections[1:].sum())
    whole_pred = int(pred_counts[1:].sum())
    whole_target = int(target_counts[1:].sum())
    whole_denom = whole_pred + whole_target

    whole_dice = (
        2 * whole_intersection / whole_denom
        if whole_denom else 1.0
    )
    precision = whole_intersection / whole_pred if whole_pred else 0.0
    recall = whole_intersection / whole_target if whole_target else 0.0

    label_metrics = {}
    for model_label in (1, 2, 3):
        denom = pred_counts[model_label] + target_counts[model_label]

        # Undefined if the reference has no voxels of this label.
        if target_counts[model_label] == 0:
            dice = None
        else:
            dice = (
                2 * intersections[model_label] / denom
                if denom else 0.0
            )

        original_label = MODEL_TO_ORIGINAL[model_label]
        label_metrics[original_label] = {
            "dice": dice,
            "predicted_voxels": int(pred_counts[model_label]),
            "reference_voxels": int(target_counts[model_label]),
        }

    return {
        "whole_dice": whole_dice,
        "precision": precision,
        "recall": recall,
        "predicted_voxels": whole_pred,
        "reference_voxels": whole_target,
        "labels": label_metrics,
    }


def main():
    loaded_models = {}

    for name, config in MODELS.items():
        path = config["path"]
        if not path.exists():
            raise FileNotFoundError(path)

        checkpoint = torch.load(
            path, map_location=DEVICE, weights_only=True
        )

        size = checkpoint.get("image_size", IMAGE_SIZE)
        if size != IMAGE_SIZE:
            raise ValueError(
                f"{name} expects {size}x{size}, "
                f"but evaluation uses {IMAGE_SIZE}x{IMAGE_SIZE}."
            )

        model = UNet(num_classes=checkpoint["num_classes"])
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(DEVICE)
        model.eval()

        loaded_models[name] = (model, config["normalization"])

    lines = [
        "Standardized U-Net v1 vs v2 evaluation",
        f"Patients: {', '.join(PATIENTS)}",
        f"Resolution: {IMAGE_SIZE} x {IMAGE_SIZE}",
        "Reference labels: 0, 1, 2, 4",
        "Internal model labels: 0, 1, 2, 3",
        "All slices included; metrics aggregate resized voxel counts.",
        "Each model uses its original training normalization.",
        "Scores are preliminary because the patient sample is small.",
    ]

    for patient in PATIENTS:
        case_id = f"BraTS20_Training_{patient}"
        volumes, reference = load_case(case_id)

        print(f"\n========== {case_id} ==========")
        lines.append(f"\n{case_id}")

        for name, (model, normalization) in loaded_models.items():
            metrics = evaluate_model(
                model, volumes, reference, normalization
            )

            print(f"\n{name} (normalization={normalization})")
            print(f"Whole-tumor Dice: {metrics['whole_dice']:.4f}")
            print(f"Precision:        {metrics['precision']:.4f}")
            print(f"Recall:           {metrics['recall']:.4f}")
            print(
                f"Predicted voxels: {metrics['predicted_voxels']}"
            )
            print(
                f"Reference voxels: {metrics['reference_voxels']}"
            )

            lines.append(
                f"{name}: whole Dice={metrics['whole_dice']:.4f}, "
                f"precision={metrics['precision']:.4f}, "
                f"recall={metrics['recall']:.4f}, "
                f"predicted voxels={metrics['predicted_voxels']}, "
                f"reference voxels={metrics['reference_voxels']}"
            )

            for label, values in metrics["labels"].items():
                score = values["dice"]
                score_text = "N/A" if score is None else f"{score:.4f}"

                message = (
                    f"  Original label {label}: Dice={score_text}, "
                    f"predicted={values['predicted_voxels']}, "
                    f"reference={values['reference_voxels']}"
                )
                print(message)
                lines.append(message)

    output = OUTPUT_PATH
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"\nSaved standardized report: {output}")
    print("Research prototype only; not for clinical use.")


if __name__ == "__main__":
    main()


