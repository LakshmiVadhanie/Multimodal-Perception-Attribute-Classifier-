"""
classification/classifier_bridge.py

Thin wrapper that connects the existing ViTAttributeClassifier to the
new detection + tracking pipeline.

Takes a raw numpy crop (from the detector) and returns structured
attribute predictions. Caches results per track_id so we only run the
heavy model every N frames (controlled by the tracker).
"""

from pathlib import Path
from typing import Dict, Optional

import numpy as np
from PIL import Image
from loguru import logger

# these imports assume the original project is on PYTHONPATH
# see integration notes in README_INTEGRATION.md
from src.serving.inference import AttributeInference
from src.serving.schemas import PredictionResponse


class ClassifierBridge:
    """
    Wraps AttributeInference for use inside the video pipeline.

    The tracker calls should_classify() to decide whether to send a crop
    here. This bridge handles PIL conversion and result caching.
    """

    def __init__(
        self,
        model_path: str,
        model_name: str = "google/vit-base-patch16-224",
        device: Optional[str] = None,
    ):
        self._inference = AttributeInference(
            model_path=model_path,
            model_name=model_name,
            device=device or self._auto_device(),
        )
        self._cache: Dict[int, PredictionResponse] = {}

    def _auto_device(self) -> str:
        try:
            import torch
            if torch.backends.mps.is_available():
                return "mps"
            if torch.cuda.is_available():
                return "cuda"
        except ImportError:
            pass
        return "cpu"

    def load(self):
        self._inference.load()

    def classify_crop(
        self,
        crop: np.ndarray,
        road_user_type: str,
        track_id: int,
    ) -> PredictionResponse:
        """
        Classify a numpy BGR crop. Caches the result by track_id.
        Returns the cached result if classification is skipped this frame.
        """
        # convert BGR numpy to PIL RGB
        pil_image = Image.fromarray(crop[..., ::-1])

        result = self._inference.predict(pil_image, road_user_type=road_user_type)
        self._cache[track_id] = result
        return result

    def get_cached(self, track_id: int) -> Optional[PredictionResponse]:
        return self._cache.get(track_id)

    def evict(self, track_id: int):
        """Call when a track is dropped so we don't leak memory."""
        self._cache.pop(track_id, None)

    def flatten_attributes(self, response: PredictionResponse) -> Dict[str, str]:
        """Returns just label strings for each attribute — useful for alert rules."""
        return {
            attr_name: pred.label
            for attr_name, pred in response.attributes.items()
        }
