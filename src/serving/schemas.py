"""
Pydantic schemas for the prediction API.
"""

from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class AttributePrediction(BaseModel):
    label: str
    confidence: float = Field(ge=0.0, le=1.0)
    probabilities: Dict[str, float]


class PredictionResponse(BaseModel):
    road_user_type: str
    attributes: Dict[str, AttributePrediction]
    inference_time_ms: float
    demo_mode: bool = False


class BatchPredictionItem(BaseModel):
    road_user_type: str = "vehicle"


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    device: str


class ErrorResponse(BaseModel):
    detail: str
