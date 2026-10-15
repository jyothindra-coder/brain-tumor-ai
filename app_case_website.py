from pathlib import Path
import json
import io
import re
import tempfile

import numpy as np
import streamlit as st
from PIL import Image

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "models"
OUTPUT_DIR = ROOT / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

st.set_page_config(
    page_title="MRI Research Assistant",
    page_icon="🧠",
    layout="wide",
)

st.markdown("""
<style>
.block-container {max-width: 1180px; padding-top: 1.5rem;}
.hero {padding: 1.7rem 1.8rem; border-radius: 18px; color: white;
       background: linear-gradient(120deg,#10243e,#23536a,#397b78);}
.hero h1 {color:white; margin-bottom:.3rem;}
.hero p {color:#e4f2f3; font-size:1.05rem;}
.notice {padding: 1rem 1.1rem; border: 1px solid #d97706; border-radius: 12px;
         background: rgba(217,119,6,.08);}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
  <h1>🧠 MRI Research Assistant</h1>
  <p>Explore an experimental brain MRI image-classification and tumor-segmentation workflow.</p>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="notice"><strong>Important medical safety notice:</strong>
This is an educational research prototype, not a medical device. It cannot diagnose or rule out a brain tumor,
and its predictions must not guide treatment or delay care. MRI interpretation must be performed by a qualified
radiologist/clinician. If you have new or severe neurological symptoms, seek urgent medical care.</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.header("Choose a workflow")
    workflow = st.radio("Workflow", ["Classify one MRI image", "Segment a 3D MRI study", "How to use safely"])
    st.divider()
    st.caption("Runs locally using model files on your computer. Uploaded files are processed by this app; don't upload scans you aren't authorized to share.")

def load_classes(path):
    if not path.exists():
        return ["glioma", "meningioma", "notumor", "pituitary"]
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        # Common format: {"0":"glioma", ...} or {"glioma":0,...}
        if all(str(k).isdigit() for k in data.keys()):
            return [data[str(i)] for i in range(len(data))]
        return [k for k, _ in sorted(data.items(), key=lambda kv: kv[1])]
    return list(data)

def get_classifier():
    import torch
    from torchvision import models
    checkpoint_path = MODEL_DIR / "brain_tumor_resnet18_v3.pth"
    classes_path = MODEL_DIR / "classes_v3.json"
    if not checkpoint_path.exists():
        checkpoint_path = MODEL_DIR / "brain_tumor_resnet18.pth"
        classes_path = MODEL_DIR / "classes.json"
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            "Classifier checkpoint not found. Expected models/brain_tumor_resnet18_v3.pth "
            "or models/brain_tumor_resnet18.pth."
        )
    classes = load_classes(classes_path)
    model = models.resnet18(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, len(classes))
    saved = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if isinstance(saved, dict) and "model_state_dict" in saved:
        saved = saved["model_state_dict"]
    model.load_state_dict(saved)
    model.eval()
    return model, classes, checkpoint_path.name

if workflow == "Classify one MRI image":
    st.subheader("Experimental image classification")
    st.write("Upload a single MRI image in PNG, JPG, or JPEG format. This workflow uses the existing ResNet18 image classifier.")
    st.warning("A normal-looking prediction does not rule out disease. A predicted class or score is not a clinical probability.")
    uploaded = st.file_uploader("Choose an MRI image", type=["png", "jpg", "jpeg"])
    if uploaded:
        image = Image.open(uploaded).convert("RGB")
        col1, col2 = st.columns([1, 1])
        with col1:
            st.image(image, caption="Uploaded image", use_container_width=True)
        with col2:
            st.markdown("#### Before running")
            st.write("This model was trained on a specific dataset and may not generalize to scans from other hospitals, scanners, sequences, or patients.")
            consent = st.checkbox("I understand this is a research demonstration, not a diagnosis.")
            if st.button("Run experimental model", type="primary", disabled=not consent):
                try:
                    import torch
                    from torchvision import transforms
                    model, classes, model_name = get_classifier()
                    transform = transforms.Compose([
                        transforms.Resize((224, 224)),
                        transforms.ToTensor(),
                        transforms.Normalize(
                            mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225],
                        ),
                    ])
                    x = transform(image).unsqueeze(0)
                    with torch.no_grad():
                        logits = model(x)
                        scores = torch.softmax(logits, dim=1)[0].cpu().numpy()
                    order = np.argsort(scores)[::-1]
                    st.markdown("#### Model output — not a diagnosis")
                    st.metric("Top predicted training category", str(classes[int(order[0])]))
                    st.caption(f"Model file: {model_name}. Displayed scores are model scores, not calibrated medical probabilities.")
                    score_table = [{"Category": str(classes[int(i)]), "Model score": float(scores[i])} for i in order]
                    st.dataframe(score_table, use_container_width=True, hide_index=True)
                    st.bar_chart({str(classes[int(i)]): float(scores[i]) for i in order})
                    st.info("Please have the original scan reviewed by a qualified radiologist. Do not make medical decisions based on this output.")
                except Exception as e:
                    st.error(f"Could not run the classifier: {e}")
                    st.write("Check that the checkpoint, class names, and dependencies match the training code.")

elif workflow == "Segment a 3D MRI study":
    st.subheader("Experimental tumor-region segmentation")
    st.write("This workflow requires one BraTS-style NIfTI file for each MRI modality: FLAIR, T1, T1ce, and T2. A single JPG/PNG image is not enough for this segmentation model.")
    st.warning("The segmentation model is a research prototype trained on a very small sample. Its colored mask is not a reliable tumor boundary.")
    uploaded_files = st.file_uploader(
        "Upload the four NIfTI modality files (.nii or .nii.gz)",
        type=["nii", "gz"], accept_multiple_files=True
    )
    st.caption("Use filenames that contain the modality names: flair, t1, t1ce, and t2. All four files must belong to the same study.")
    confirmed = st.checkbox("I understand that segmentation output is experimental and requires specialist review.")
    if st.button("Run experimental segmentation", type="primary", disabled=not confirmed):
        if len(uploaded_files) != 4:
            st.error("Please upload exactly four files: FLAIR, T1, T1ce, and T2.")
        else:
            try:
                import nibabel as nib
                import torch
                import torch.nn.functional as F
                import matplotlib.pyplot as plt
                from train_unet import UNet

                files_by_modality = {}
                for f in uploaded_files:
                    name = f.name.lower()
                    matches = [m for m in ("flair", "t1ce", "t1", "t2") if m in name]
                    if not matches:
                        raise ValueError(f"Cannot identify modality from filename: {f.name}")
                    modality = matches[0]
                    if modality in files_by_modality:
                        raise ValueError(f"More than one file matched {modality}.")
                    files_by_modality[modality] = f
                required = ["flair", "t1", "t1ce", "t2"]
                if set(files_by_modality) != set(required):
                    raise ValueError("Could not identify all four modalities from filenames.")

                arrays = []
                affines = []
                for modality in required:
                    f = files_by_modality[modality]
                    # nibabel loads from a path, so write the uploaded bytes to a temporary file.
                    suffix = ".nii.gz" if f.name.lower().endswith(".nii.gz") else ".nii"
                    with tempfile.NamedTemporaryFile(suffix=suffix) as tmp:
                        tmp.write(f.getvalue())
                        tmp.flush()
                        nii = nib.load(tmp.name)
                        arr = np.asarray(nii.dataobj, dtype=np.float32)
                        affine = nii.affine.copy()
                    if arr.ndim != 3:
                        raise ValueError(f"{f.name} is not a 3D volume.")
                    arrays.append(arr)
                    affines.append(affine)
                shape = arrays[0].shape
                if any(a.shape != shape for a in arrays):
                    raise ValueError("The four volumes have different dimensions. Confirm they belong to the same aligned study.")
                checkpoint_path = MODEL_DIR / "brats_unet_v2.pth"
                if not checkpoint_path.exists():
                    checkpoint_path = MODEL_DIR / "brats_unet_sample.pth"
                if not checkpoint_path.exists():
                    raise FileNotFoundError("No segmentation checkpoint found in models/.")
                checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
                model = UNet(num_classes=4)
                model.load_state_dict(checkpoint["model_state_dict"])
                model.eval()
                size = int(checkpoint.get("image_size", 128))

                normed = []
                for a in arrays:
                    a = a.copy()
                    brain = a != 0
                    if np.any(brain):
                        a[brain] = (a[brain] - a[brain].mean()) / max(float(a[brain].std()), 1e-6)
                    normed.append(a)

                z = st.slider("Select axial slice", 0, shape[2]-1, shape[2]//2)
                channels = []
                for a in normed:
                    t = torch.from_numpy(a[:, :, z].copy())[None, None]
                    t = F.interpolate(t, size=(size, size), mode="bilinear", align_corners=False)[0, 0]
                    channels.append(t)
                x = torch.stack(channels)[None]
                with torch.no_grad():
                    pred = model(x).argmax(dim=1)[0].numpy()
                pred_original = np.choose(pred, [0, 1, 2, 4])
                flair = F.interpolate(torch.from_numpy(normed[0][:, :, z].copy())[None, None], size=(size, size), mode="bilinear", align_corners=False)[0, 0].numpy()
                fig, axes = plt.subplots(1, 2, figsize=(10, 5))
                axes[0].imshow(flair.T, cmap="gray", origin="lower")
                axes[0].set_title("FLAIR MRI")
                axes[1].imshow(flair.T, cmap="gray", origin="lower")
                overlay = np.ma.masked_where(pred_original.T == 0, pred_original.T)
                axes[1].imshow(overlay, cmap="jet", vmin=1, vmax=4, alpha=.55, origin="lower", interpolation="nearest")
                axes[1].set_title("Experimental predicted mask")
                for ax in axes:
                    ax.axis("off")
                fig.tight_layout()
                st.pyplot(fig)
                plt.close(fig)
                st.caption(f"Checkpoint: {checkpoint_path.name}. Internal labels are displayed as BraTS labels 1, 2, and 4. No clinical interpretation is provided.")
                st.info("This model was trained on only a small sample and has not been validated for clinical use. Have MRI findings interpreted by a radiologist.")
            except Exception as e:
                st.error(f"Could not run segmentation: {e}")
                st.write("Check file alignment, modality filenames, checkpoint files, and installed dependencies.")

else:
    st.subheader("How to use this safely")
    st.markdown("""
1. Use only MRI files you are authorized to process.
2. For image classification, upload a supported single image.
3. For segmentation, upload all four aligned 3D NIfTI modalities from the same study.
4. Treat every result as an experimental model output, not a diagnosis.
5. Share the original scan and report with a qualified radiologist or neurologist.
""")
    st.markdown("### Seek urgent medical care if you have")
    st.write("Sudden weakness, new seizures, confusion, severe sudden headache, difficulty speaking, or sudden vision changes. Do not wait for an AI result.")
    st.markdown("### What this prototype cannot do")
    st.write("It cannot diagnose cancer, determine tumor grade, replace radiology interpretation, reliably rule out a tumor, or recommend treatment.")

st.divider()
st.caption("MRI Research Assistant • Local research prototype • Not for clinical use")
