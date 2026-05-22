"""
Inference logic for the attribute classifier.

Wraps model loading, preprocessing, and postprocessing into a clean
interface used by the FastAPI app.

When no trained checkpoint exists the class falls back to mock_predict(),
which produces plausible random probabilities for demo purposes.
"""

import random
import time
from pathlib import Path
from typing import Dict, Optional

import torch
from PIL import Image
from loguru import logger

from src.dataset.augmentations import build_inference_transforms
from src.dataset.ontology import LABEL_MAPS, ONTOLOGY, get_applicable_attributes
from src.models.vit_classifier import ViTAttributeClassifier
from src.serving.schemas import AttributePrediction, PredictionResponse


class AttributeInference:
    """
    Manages model loading and runs inference on single images.
    Designed to be instantiated once and reused across requests.

    Falls back to mock_predict() when no trained checkpoint is available
    so the API stays fully functional without model weights.
    """

    def __init__(
        self,
        model_path: str,
        model_name: str = "google/vit-base-patch16-224",
        device: Optional[str] = None,
        image_size: int = 224,
    ):
        self.model_path = model_path
        self.model_name = model_name
        self.image_size = image_size
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.model: Optional[ViTAttributeClassifier] = None
        self.transform = build_inference_transforms(image_size)
        self._demo_mode: bool = False

    def load(self):
        checkpoint_file = Path(self.model_path) / "model.pt"
        if not checkpoint_file.exists():
            logger.warning(
                f"No checkpoint found at {self.model_path}. "
                "Running in DEMO mode — predictions are synthetic."
            )
            self._demo_mode = True
            return

        logger.info(f"Loading model from {self.model_path} on {self.device}")
        try:
            self.model = ViTAttributeClassifier.load(
                path=self.model_path,
                model_name=self.model_name,
                device=self.device,
            )
            self.model.to(self.device)
            self.model.eval()
            self._demo_mode = False
            logger.info("Model ready")
        except Exception as exc:
            logger.warning(f"Could not load model ({exc}). Running in DEMO mode.")
            self.model = None
            self._demo_mode = True

    def is_loaded(self) -> bool:
        return self.model is not None

    @property
    def demo_mode(self) -> bool:
        return self._demo_mode

    # ------------------------------------------------------------------
    # Mock inference (demo mode)
    # ------------------------------------------------------------------

    def mock_predict(self, road_user_type: str = "vehicle") -> PredictionResponse:
        """
        Returns plausible random probabilities for all applicable attributes.
        Simulates ~90 ms latency so the loading animation looks realistic.
        """
        t0 = time.perf_counter()
        time.sleep(random.uniform(0.07, 0.14))  # realistic latency

        applicable = set(get_applicable_attributes(road_user_type))
        attribute_predictions: Dict[str, AttributePrediction] = {}

        for attr_name in applicable:
            label_map = LABEL_MAPS[attr_name]
            num_classes = len(label_map)
            raw = [random.uniform(0.05, 1.0) for _ in range(num_classes)]
            total = sum(raw)
            probs = [v / total for v in raw]

            # bias the highest prob class so results look confident
            best_idx = probs.index(max(probs))
            probs[best_idx] = min(probs[best_idx] * 1.8, 0.95)
            total2 = sum(probs)
            probs = [v / total2 for v in probs]

            confidence = probs[best_idx]
            best_label = label_map[best_idx]
            prob_dict = {label_map[i]: probs[i] for i in range(num_classes)}

            attribute_predictions[attr_name] = AttributePrediction(
                label=best_label,
                confidence=confidence,
                probabilities=prob_dict,
            )

        elapsed_ms = (time.perf_counter() - t0) * 1000
        return PredictionResponse(
            road_user_type=road_user_type,
            attributes=attribute_predictions,
            inference_time_ms=round(elapsed_ms, 2),
            demo_mode=True,
        )

    # ------------------------------------------------------------------
    # Real inference
    # ------------------------------------------------------------------

    def predict(self, image: Image.Image, road_user_type: str = "vehicle") -> PredictionResponse:
        if not self.is_loaded():
            return self.mock_predict(road_user_type)

        t0 = time.perf_counter()

        tensor = self.transform(image).unsqueeze(0).to(self.device)
        probabilities = self.model.predict(tensor)

        applicable = set(get_applicable_attributes(road_user_type))
        attribute_predictions = {}

        for attr_name, probs in probabilities.items():
            if attr_name not in applicable:
                continue

            probs_cpu = probs.squeeze(0).cpu()
            label_map = LABEL_MAPS[attr_name]
            best_idx = probs_cpu.argmax().item()
            best_label = label_map[best_idx]
            confidence = probs_cpu[best_idx].item()

            prob_dict = {
                label_map[i]: probs_cpu[i].item()
                for i in range(len(label_map))
            }

            attribute_predictions[attr_name] = AttributePrediction(
                label=best_label,
                confidence=confidence,
                probabilities=prob_dict,
            )

        elapsed_ms = (time.perf_counter() - t0) * 1000

        return PredictionResponse(
            road_user_type=road_user_type,
            attributes=attribute_predictions,
            inference_time_ms=round(elapsed_ms, 2),
            demo_mode=False,
        )
