"""
detection/detector.py

YOLOv8-based road user detector.
Takes a raw frame and returns bounding boxes + class labels for
vehicles, pedestrians, and cyclists.

Install dependency:
    pip install ultralytics
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from loguru import logger


# COCO class IDs that map to our three road user types
COCO_TO_ROAD_USER = {
    0: "pedestrian",   # person
    1: "cyclist",      # bicycle
    2: "vehicle",      # car
    3: "vehicle",      # motorcycle (close enough for attribute classification)
    5: "vehicle",      # bus
    7: "vehicle",      # truck
}


@dataclass
class Detection:
    bbox: Tuple[int, int, int, int]   # x1, y1, x2, y2 in pixel coords
    road_user_type: str               # vehicle | pedestrian | cyclist
    confidence: float
    track_id: Optional[int] = None   # filled in by tracker


@dataclass
class DetectionFrame:
    frame_idx: int
    timestamp_ms: float
    detections: List[Detection] = field(default_factory=list)


class RoadUserDetector:
    """
    Wraps YOLOv8 to detect road users in a single frame.

    Uses the pretrained COCO weights by default. For better accuracy on
    traffic footage, fine-tune on BDD100K or nuScenes and pass the
    custom weights path.
    """

    def __init__(
        self,
        weights: str = "yolov8n.pt",   # nano = fastest; use yolov8m.pt for better accuracy
        confidence_threshold: float = 0.4,
        device: Optional[str] = None,
        min_bbox_area: int = 400,       # pixels^2, filters out tiny far-away detections
    ):
        self.weights = weights
        self.confidence_threshold = confidence_threshold
        self.min_bbox_area = min_bbox_area
        self.device = device or self._auto_device()
        self._model = None

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
        from ultralytics import YOLO
        logger.info(f"Loading YOLO weights: {self.weights} on {self.device}")
        self._model = YOLO(self.weights)
        logger.info("Detector ready")

    def detect(self, frame: np.ndarray, frame_idx: int = 0, timestamp_ms: float = 0.0) -> DetectionFrame:
        """
        Run detection on a single BGR or RGB numpy frame.
        Returns a DetectionFrame with all road user detections.
        """
        if self._model is None:
            raise RuntimeError("Call load() before detecting")

        results = self._model(
            frame,
            conf=self.confidence_threshold,
            device=self.device,
            verbose=False,
        )[0]

        detections = []
        for box in results.boxes:
            class_id = int(box.cls.item())
            if class_id not in COCO_TO_ROAD_USER:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            area = (x2 - x1) * (y2 - y1)
            if area < self.min_bbox_area:
                continue

            detections.append(Detection(
                bbox=(x1, y1, x2, y2),
                road_user_type=COCO_TO_ROAD_USER[class_id],
                confidence=float(box.conf.item()),
            ))

        return DetectionFrame(
            frame_idx=frame_idx,
            timestamp_ms=timestamp_ms,
            detections=detections,
        )

    def crop(self, frame: np.ndarray, detection: Detection) -> np.ndarray:
        """Crop the frame to the detection bounding box with a small padding."""
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = detection.bbox
        pad = 10
        x1 = max(0, x1 - pad)
        y1 = max(0, y1 - pad)
        x2 = min(w, x2 + pad)
        y2 = min(h, y2 + pad)
        return frame[y1:y2, x1:x2]
