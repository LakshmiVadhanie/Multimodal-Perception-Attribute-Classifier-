"""
FastAPI application for serving road user attribute predictions.

Startup is fully graceful — if no trained checkpoint exists the application
runs in demo mode and returns synthetic (but realistic) predictions so that
the frontend can be exercised without model weights.
"""

import io
import os
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger
from PIL import Image

from src.serving.inference import AttributeInference
from src.serving.schemas import HealthResponse, PredictionResponse
from src.dataset.ontology import ROAD_USER_TYPES


MODEL_PATH = os.getenv("MODEL_PATH", "./checkpoints/best_model")
MODEL_NAME = os.getenv("MODEL_NAME", "google/vit-base-patch16-224")
DEVICE = os.getenv("DEVICE", "cpu")

predictor = AttributeInference(
    model_path=MODEL_PATH,
    model_name=MODEL_NAME,
    device=DEVICE,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    predictor.load()          # never raises — falls back to demo mode
    yield


app = FastAPI(
    title="Road User Attribute Classifier",
    description=(
        "Classifies behavioral and physical attributes of road users "
        "(vehicles, pedestrians, cyclists) from images. "
        "Runs in demo mode when no trained checkpoint is present."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(
        status="ok",
        model_loaded=predictor.is_loaded(),
        device=predictor.device,
    )


@app.get("/demo")
def demo_status():
    """Returns whether the API is operating in demo (mock) mode."""
    return {"demo_mode": predictor.demo_mode, "model_loaded": predictor.is_loaded()}


@app.post("/predict", response_model=PredictionResponse)
async def predict(
    file: UploadFile = File(..., description="Image of a road user (JPEG or PNG)"),
    road_user_type: str = Form(
        default="vehicle",
        description=f"Type of road user. One of: {ROAD_USER_TYPES}",
    ),
):
    if road_user_type not in ROAD_USER_TYPES:
        raise HTTPException(
            status_code=422,
            detail=f"Invalid road_user_type '{road_user_type}'. Must be one of {ROAD_USER_TYPES}.",
        )

    if file.content_type not in ("image/jpeg", "image/png", "image/webp"):
        raise HTTPException(
            status_code=415,
            detail="Unsupported media type. Upload a JPEG, PNG, or WEBP image.",
        )

    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not read image: {e}")

    try:
        result = predictor.predict(image, road_user_type=road_user_type)
    except Exception as e:
        logger.exception("Prediction failed")
        raise HTTPException(status_code=500, detail=f"Prediction error: {e}")

    return result
