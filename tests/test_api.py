"""Tests for the FastAPI prediction endpoints."""

import io
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from src.serving.schemas import AttributePrediction, PredictionResponse


def _make_fake_response():
    return PredictionResponse(
        road_user_type="vehicle",
        attributes={
            "mobility": AttributePrediction(
                label="moving",
                confidence=0.92,
                probabilities={"moving": 0.92, "stationary": 0.05, "slow_moving": 0.03},
            ),
            "occlusion": AttributePrediction(
                label="none",
                confidence=0.88,
                probabilities={"none": 0.88, "partial": 0.10, "heavy": 0.02},
            ),
        },
        inference_time_ms=14.5,
    )


def _make_image_bytes(size=(100, 100), fmt="JPEG"):
    img = Image.new("RGB", size, color=(128, 64, 32))
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    buf.seek(0)
    return buf.read()


@pytest.fixture
def client():
    # patch the predictor so we do not load real model weights in tests
    with patch("src.serving.api.predictor") as mock_predictor:
        mock_predictor.is_loaded.return_value = True
        mock_predictor.device = "cpu"
        mock_predictor.predict.return_value = _make_fake_response()

        from src.serving.api import app
        yield TestClient(app)


class TestHealthEndpoint:
    def test_health_ok(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert data["model_loaded"] is True

    def test_health_schema(self, client):
        response = client.get("/health")
        data = response.json()
        assert "status" in data
        assert "model_loaded" in data
        assert "device" in data


class TestPredictEndpoint:
    def test_predict_vehicle(self, client):
        image_bytes = _make_image_bytes()
        response = client.post(
            "/predict",
            files={"file": ("test.jpg", image_bytes, "image/jpeg")},
            data={"road_user_type": "vehicle"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["road_user_type"] == "vehicle"
        assert "attributes" in data
        assert "inference_time_ms" in data

    def test_predict_pedestrian(self, client):
        image_bytes = _make_image_bytes()
        response = client.post(
            "/predict",
            files={"file": ("test.jpg", image_bytes, "image/jpeg")},
            data={"road_user_type": "pedestrian"},
        )
        assert response.status_code == 200

    def test_invalid_road_user_type(self, client):
        image_bytes = _make_image_bytes()
        response = client.post(
            "/predict",
            files={"file": ("test.jpg", image_bytes, "image/jpeg")},
            data={"road_user_type": "airplane"},
        )
        assert response.status_code == 422

    def test_unsupported_media_type(self, client):
        response = client.post(
            "/predict",
            files={"file": ("test.txt", b"not an image", "text/plain")},
            data={"road_user_type": "vehicle"},
        )
        assert response.status_code == 415

    def test_attribute_prediction_schema(self, client):
        image_bytes = _make_image_bytes()
        response = client.post(
            "/predict",
            files={"file": ("test.jpg", image_bytes, "image/jpeg")},
            data={"road_user_type": "vehicle"},
        )
        data = response.json()
        for attr_name, attr_pred in data["attributes"].items():
            assert "label" in attr_pred
            assert "confidence" in attr_pred
            assert "probabilities" in attr_pred
            assert 0.0 <= attr_pred["confidence"] <= 1.0

    def test_png_accepted(self, client):
        image_bytes = _make_image_bytes(fmt="PNG")
        response = client.post(
            "/predict",
            files={"file": ("test.png", image_bytes, "image/png")},
            data={"road_user_type": "vehicle"},
        )
        assert response.status_code == 200
