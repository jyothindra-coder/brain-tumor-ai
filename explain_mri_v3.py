
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn
from torchvision import models, transforms
from PIL import Image, ImageDraw, ImageOps


# --------------------------------------------------
# Configuration
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"
OUTPUT_DIR = BASE_DIR / "outputs"

MODEL_PATH = MODEL_DIR / "brain_tumor_resnet18_v3.pth"
CLASSES_PATH = MODEL_DIR / "classes_v3.json"

DEVICE = torch.device("cpu")
IMAGE_SIZE = 224

PREPROCESS = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


# --------------------------------------------------
# Load the existing trained model
# --------------------------------------------------

def load_model():
    if not MODEL_PATH.is_file() or not CLASSES_PATH.is_file():
        raise FileNotFoundError(
            "Model files are missing. Check the models folder."
        )

    with open(CLASSES_PATH, "r", encoding="utf-8") as file:
        classes = json.load(file)

    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(classes))

    state = torch.load(
        MODEL_PATH,
        map_location=DEVICE,
        weights_only=True,
    )

    model.load_state_dict(state)
    model.eval()

    return model, classes


# --------------------------------------------------
# Generate Grad-CAM
# --------------------------------------------------

def explain_image(image_path):
    image_path = Path(image_path)

    if not image_path.is_file():
        raise FileNotFoundError(f"Image not found: {image_path}")

    with Image.open(image_path) as source:
        original_image = source.convert("RGB")

    model, classes = load_model()

    # ResNet18's final convolutional block
    target_layer = model.layer4[-1]
    captured = {}

    def save_activation(module, inputs, output):
        captured["activation"] = output
        output.retain_grad()

    hook = target_layer.register_forward_hook(save_activation)

    try:
        tensor = PREPROCESS(original_image).unsqueeze(0)

        # Gradients are required for Grad-CAM.
        model.zero_grad(set_to_none=True)

        with torch.enable_grad():
            logits = model(tensor)
            scores = torch.softmax(logits.detach(), dim=1)[0]

            predicted_index = int(scores.argmax())
            predicted_class = classes[predicted_index]

            logits[0, predicted_index].backward()

        activation = captured["activation"].detach()
        gradient = captured["activation"].grad.detach()

        # Average gradients over the spatial dimensions.
        weights = gradient.mean(dim=(2, 3), keepdim=True)

        # Weighted combination of feature maps.
        cam = (weights * activation).sum(dim=1, keepdim=True)
        cam = F.relu(cam)

        cam = F.interpolate(
            cam,
            size=(IMAGE_SIZE, IMAGE_SIZE),
            mode="bilinear",
            align_corners=False,
        )[0, 0]

        cam = cam - cam.min()
        cam = cam / cam.max().clamp_min(1e-8)
        cam = cam.detach().cpu()

    finally:
        hook.remove()

    # Convert the heatmap into a PIL grayscale image.
    cam_bytes = (cam * 255).byte().contiguous().numpy().tobytes()
    gray_heatmap = Image.frombytes(
        "L",
        (IMAGE_SIZE, IMAGE_SIZE),
        cam_bytes,
    )

    # Dark blue -> yellow -> red indicates increasing attention.
    color_heatmap = ImageOps.colorize(
        gray_heatmap,
        black="#101040",
        white="#ff0000",
        blackpoint=0,
        whitepoint=255,
    )

    # Resize the heatmap to match the original MRI.
    color_heatmap = color_heatmap.resize(
        original_image.size,
        Image.Resampling.BILINEAR,
    )

    overlay = Image.blend(
        original_image,
        color_heatmap,
        alpha=0.40,
    )

    # Find the strongest attention point and an approximate
    # high-attention region. This is NOT a tumor segmentation mask.
    mask = cam >= 0.60
    locations = torch.nonzero(mask, as_tuple=False)

    width, height = original_image.size
    draw = ImageDraw.Draw(overlay)

    max_y, max_x = torch.unravel_index(
        cam.argmax(),
        cam.shape,
    )
    marker_x = int(max_x.item() * width / IMAGE_SIZE)
    marker_y = int(max_y.item() * height / IMAGE_SIZE)

    # Mark the strongest attention point.
    radius = max(5, min(width, height) // 40)
    draw.ellipse(
        (
            marker_x - radius,
            marker_y - radius,
            marker_x + radius,
            marker_y + radius,
        ),
        outline="white",
        width=3,
    )

    # Outline the approximate high-attention area.
    if len(locations) > 0:
        y_min = int(locations[:, 0].min().item() * height / IMAGE_SIZE)
        y_max = int((locations[:, 0].max().item() + 1) * height / IMAGE_SIZE)
        x_min = int(locations[:, 1].min().item() * width / IMAGE_SIZE)
        x_max = int((locations[:, 1].max().item() + 1) * width / IMAGE_SIZE)

        draw.rectangle(
            (x_min, y_min, x_max, y_max),
            outline="white",
            width=3,
        )
    else:
        x_min = y_min = x_max = y_max = None

    # --------------------------------------------------
    # Save the highlighted image and text report
    # --------------------------------------------------

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    image_output = OUTPUT_DIR / f"{image_path.stem}_gradcam.png"
    report_output = OUTPUT_DIR / f"{image_path.stem}_report.txt"

    overlay.save(image_output)

    score_lines = [
        f"{name}: {score.item():.2%}"
        for name, score in zip(classes, scores)
    ]

    report = f"""
BRAIN MRI AI — EDUCATIONAL ANALYSIS REPORT
==========================================

Input image: {image_path.name}

Predicted dataset class: {predicted_class}
Highest model score: {scores[predicted_index].item():.2%}

MODEL SCORES
------------
{chr(10).join(score_lines)}

REGION HIGHLIGHTING
-------------------
Method: Grad-CAM on ResNet18 layer4
Marked point: ({marker_x}, {marker_y}) in original image pixels
Approximate high-attention box: {x_min}, {y_min}, {x_max}, {y_max}

Interpretation:
The heatmap indicates image regions that influenced the
model's selected class. It does not confirm the presence,
location, size, or boundary of a tumor.

OUTPUT FILES
------------
Highlighted image: {image_output}
This report: {report_output}

IMPORTANT LIMITATIONS
---------------------
This is an educational research prototype, not a validated
medical diagnostic system. Model scores are not calibrated
clinical probabilities. The highlighted region is not a
verified tumor segmentation. Results must not be used to
make medical decisions.
"""

    report_output.write_text(report.strip() + "\n", encoding="utf-8")

    print("\n--- BRAIN MRI AI REPORT ---")
    print(f"Predicted class: {predicted_class}")
    print(f"Highest model score: {scores[predicted_index].item():.2%}")

    print("\nClass scores:")
    for line in score_lines:
        print(line)

    print("\nSaved highlighted image:", image_output)
    print("Saved report:", report_output)
    print("\nEducational prototype only; not for medical diagnosis.")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(
            'Usage: python explain_mri.py "path/to/mri_image.jpg"'
        )
        sys.exit(1)

    try:
        explain_image(sys.argv[1])
    except Exception as error:
        print(f"Unable to analyze image: {error}")
        sys.exit(1)
