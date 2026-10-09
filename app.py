
import json
from pathlib import Path

import streamlit as st
import torch
from torch import nn
from torchvision import models, transforms
from PIL import Image

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"

st.set_page_config(
    page_title="Brain MRI AI",
    page_icon="🧠",
    layout="centered",
)

st.title("🧠 Brain MRI Classification")
st.write(
    "An educational deep learning project using PyTorch "
    "and ResNet18."
)

st.warning(
    "Research prototype only. This application is not "
    "validated for medical diagnosis or clinical decisions."
)

@st.cache_resource
def load_model():
    with open(MODEL_DIR / "classes.json", "r") as file:
        classes = json.load(file)

    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(classes))

    model.load_state_dict(
        torch.load(
            MODEL_DIR / "brain_tumor_resnet18.pth",
            map_location="cpu",
            weights_only=True,
        )
    )

    model.eval()
    return model, classes


preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])

uploaded_file = st.file_uploader(
    "Upload a brain MRI image",
    type=["jpg", "jpeg", "png"],
)

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded MRI image", width=350)

        if st.button("Analyze MRI image", type="primary"):
            model, classes = load_model()
            tensor = preprocess(image).unsqueeze(0)

            with torch.no_grad():
                logits = model(tensor)
                scores = torch.softmax(logits, dim=1)[0]

            index = int(scores.argmax())

            st.subheader("Model output")
            st.write(f"Predicted class: **{classes[index]}**")
            st.metric(
                "Highest model score",
                f"{scores[index].item():.2%}",
            )

            st.subheader("Scores by class")

            for name, score in zip(classes, scores.tolist()):
                st.write(f"**{name}:** {score:.2%}")
                st.progress(float(score))

            st.caption(
                "Scores are model outputs, not calibrated medical "
                "probabilities. Do not use them for diagnosis."
            )

    except Exception as error:
        st.error(f"Unable to process the image: {error}")
