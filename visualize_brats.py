
from pathlib import Path

import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt

case_dir = Path(
    r"G:\BraTS2020_Sample\BraTS20_Training_001"
)

case_id = "BraTS20_Training_001"

flair = np.asarray(
    nib.load(str(case_dir / f"{case_id}_flair.nii")).dataobj
)

seg = np.asarray(
    nib.load(str(case_dir / f"{case_id}_seg.nii")).dataobj
)

# Select the slice containing the largest tumor area.
tumor_area = np.count_nonzero(seg > 0, axis=(0, 1))
slice_index = int(np.argmax(tumor_area))

mri_slice = flair[:, :, slice_index]
mask_slice = seg[:, :, slice_index]

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.imshow(mri_slice.T, cmap="gray", origin="lower")
plt.title(f"FLAIR MRI — Slice {slice_index}")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(mri_slice.T, cmap="gray", origin="lower")
plt.imshow(
    np.ma.masked_where(mask_slice.T == 0, mask_slice.T),
    cmap="jet",
    alpha=0.55,
    origin="lower",
    vmin=1,
    vmax=4,
    interpolation="nearest",
)
plt.title("MRI with Reference Tumor Mask")
plt.axis("off")

plt.tight_layout()
plt.savefig("outputs/brats_sample_visualization.png", dpi=150)
plt.show()

print("Slice index:", slice_index)
print("Tumor voxels on selected slice:", int(tumor_area[slice_index]))
print("Saved: outputs/brats_sample_visualization.png")
