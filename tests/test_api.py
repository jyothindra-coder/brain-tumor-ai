
from io import BytesIO

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from api import app

client = TestClient(app)


def make_png():
    """Create a small valid PNG image in memory."""
    image = Image.new("RGB", (64, 64), color="gray")
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def test_home_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["docs"] == "/docs"


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert response.json()["model_loaded"] is True


def test_prediction_with_valid_png():
    response = client.post(
        "/predict",
        files={
            "file": (
                "sample.png",
                make_png(),
                "image/png",
            )
        },
    )

    assert response.status_code == 200

    body = response.json()
    assert body["predicted_class"] in {
        "glioma",
        "meningioma",
        "notumor",
        "pituitary",
    }
    assert 0.0 <= body["model_score"] <= 1.0
    assert set(body["scores"]) == {
        "glioma",
        "meningioma",
        "notumor",
        "pituitary",
    }
    assert abs(sum(body["scores"].values()) - 1.0) < 0.001
    assert "not for medical diagnosis" in body["disclaimer"].lower()


def test_rejects_unsupported_file_type():
    response = client.post(
        "/predict",
        files={
            "file": (
                "sample.txt",
                b"not an image",
                "text/plain",
            )
        },
    )

    assert response.status_code == 415


def test_rejects_corrupted_image():
    response = client.post(
        "/predict",
        files={
            "file": (
                "broken.png",
                b"this is not a valid image",
                "image/png",
            )
        },
    )

    assert response.status_code == 400


def test_rejects_oversized_file():
    # The API accepts files up to 10 MB.
    oversized_content = b"x" * (10 * 1024 * 1024 + 1)

    response = client.post(
        "/predict",
        files={
            "file": (
                "large.png",
                oversized_content,
                "image/png",
            )
        },
    )

    assert response.status_code == 413
