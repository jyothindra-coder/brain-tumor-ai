
from pathlib import Path
import nibabel as nib
import numpy as np

case_dir = Path(r"G:\BraTS2020_Sample\BraTS20_Training_001")

modalities = ["flair", "t1", "t1ce", "t2", "seg"]

for modality in modalities:
    file_path = case_dir / f"BraTS20_Training_001_{modality}.nii"

    if not file_path.exists():
        print(f"Missing file: {file_path}")
        continue

    image = nib.load(str(file_path))
    volume = np.asarray(image.dataobj)

    print(f"\n{modality.upper()}")
    print("Shape:", volume.shape)
    print("Data type:", volume.dtype)
    print("Minimum:", float(volume.min()))
    print("Maximum:", float(volume.max()))

    if modality == "seg":
        labels, counts = np.unique(volume, return_counts=True)
        print("Segmentation labels and voxel counts:")
        print(dict(zip(labels.tolist(), counts.tolist())))
