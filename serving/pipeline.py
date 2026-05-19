"""
serving/pipeline.py

Orchestrator that wires together:
    VideoIngestor -> RoadUserDetector -> IoUTracker -> ClassifierBridge -> RuleEngine

This is the single entry point for both video file analysis and live stream processing.
The FastAPI app in serving/api_v2.py calls run_on_file() and run_on_stream().
"""

import json
import time
from pathlib import Path
from typing import Callable, Dict, Generator, List, Optional, Union

import cv2
import numpy as np
from loguru import logger

from detection.detector import RoadUserDetector
from tracking.tracker import IoUTracker
from classification.classifier_bridge import ClassifierBridge
from ingestion.video_ingestor import VideoIngestor, FrameInfo
from alerts.rule_engine import Alert, RuleEngine


class AnalysisResult:
    """Accumulates results from a full video analysis run."""

    def __init__(self, source: str):
        self.source = source
        self.total_frames_processed = 0
        self.total_detections = 0
        self.total_classifications = 0
        self.alerts: List[Dict] = []
        self.track_summaries: Dict[int, Dict] = {}
        self.started_at = time.time()
        self.finished_at: Optional[float] = None

    def finalize(self):
        self.finished_at = time.time()

    @property
    def duration_s(self) -> float:
        return round((self.finished_at or time.time()) - self.started_at, 2)

    def to_dict(self) -> Dict:
        return {
            "source": self.source,
            "total_frames_processed": self.total_frames_processed,
            "total_detections": self.total_detections,
            "total_classifications": self.total_classifications,
            "total_alerts": len(self.alerts),
            "alerts": self.alerts,
            "track_summaries": [
                ts for ts in self.track_summaries.values() 
                if ts["age_frames"] >= 3 and ts.get("last_attributes") is not None
            ],
            "duration_s": self.duration_s,
        }

    def save(self, output_path: str):
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        logger.info(f"Results saved to {output_path}")


class VideoPipeline:
    """
    Full analysis pipeline for road user attribute detection.

    Usage:
        pipeline = VideoPipeline(model_path="./checkpoints/best_model")
        pipeline.load()

        # analyze a file
        result = pipeline.run_on_file("traffic.mp4")

        # or stream live results frame by frame
        for frame_result in pipeline.stream_results("rtsp://camera-ip/stream"):
            print(frame_result)
    """

    def __init__(
        self,
        model_path: str,
        model_name: str = "google/vit-base-patch16-224",
        yolo_weights: str = "yolov8n.pt",
        detection_confidence: float = 0.4,
        iou_threshold: float = 0.20,
        classify_every_n_frames: int = 5,
        process_every_n_frames: int = 3,
        device: Optional[str] = None,
        alert_rules: Optional[List[Dict]] = None,
    ):
        self.ingestor = VideoIngestor(
            process_every_n_frames=process_every_n_frames,
        )
        self.detector = RoadUserDetector(
            weights=yolo_weights,
            confidence_threshold=detection_confidence,
            device=device,
        )
        self.tracker = IoUTracker(
            iou_threshold=iou_threshold,
            classify_every_n_frames=classify_every_n_frames,
        )
        self.classifier = ClassifierBridge(
            model_path=model_path,
            model_name=model_name,
            device=device,
        )
        self.rule_engine = RuleEngine(rules=alert_rules)
        self._loaded = False

    def load(self):
        self.detector.load()
        self.classifier.load()
        self._loaded = True
        logger.info("Pipeline ready")

    def run_on_file(
        self,
        video_path: Union[str, Path],
        output_path: Optional[str] = None,
        on_alert: Optional[Callable[[Alert], None]] = None,
    ) -> AnalysisResult:
        """
        Analyze a full video file. Blocks until complete.
        Returns an AnalysisResult with all detections, attributes, and alerts.
        """
        result = AnalysisResult(source=str(video_path))

        for _ in self._process_frames(
            self.ingestor.stream_frames(video_path),
            result=result,
            on_alert=on_alert,
        ):
            pass

        result.finalize()
        if output_path:
            result.save(output_path)

        logger.info(
            f"Analysis complete: {result.total_frames_processed} frames, "
            f"{result.total_detections} detections, "
            f"{len(result.alerts)} alerts in {result.duration_s}s"
        )
        return result

    def run_on_image(self, image_path: Union[str, Path]) -> Dict:
        """
        Analyze a single image. Returns a flat dict of detections + attributes.
        Used by the upload endpoint in the API.
        """
        frame_info = self.ingestor.load_image(image_path)
        det_frame = self.detector.detect(
            frame_info.frame,
            frame_idx=0,
            timestamp_ms=0.0,
        )

        detections_out = []
        for det in det_frame.detections:
            crop = self.detector.crop(frame_info.frame, det)
            pred = self.classifier.classify_crop(
                crop, det.road_user_type, track_id=0
            )
            attrs = self.classifier.flatten_attributes(pred)
            alerts = self.rule_engine.evaluate(
                track_id=0,
                road_user_type=det.road_user_type,
                attributes=attrs,
                frame_idx=0,
                timestamp_ms=0.0,
                bbox=det.bbox,
            )
            detections_out.append({
                "road_user_type": det.road_user_type,
                "confidence": round(det.confidence, 3),
                "bbox": list(det.bbox),
                "attributes": attrs,
                "alerts": [a.to_dict() for a in alerts],
            })

        return {
            "source": str(image_path),
            "detections": detections_out,
            "total_alerts": sum(len(d["alerts"]) for d in detections_out),
        }

    def stream_results(
        self, source: Union[str, Path]
    ) -> Generator[Dict, None, None]:
        """
        Generator for live stream processing.
        Yields a dict for each processed frame — use this for SSE or WebSocket push.
        """
        result = AnalysisResult(source=str(source))
        for frame_result in self._process_frames(
            self.ingestor.stream_frames(source),
            result=result,
        ):
            yield frame_result

    def _process_frames(
        self,
        frame_gen,
        result: AnalysisResult,
        on_alert: Optional[Callable] = None,
    ) -> Generator[Dict, None, None]:
        for frame_info in frame_gen:
            det_frame = self.detector.detect(
                frame_info.frame,
                frame_idx=frame_info.frame_idx,
                timestamp_ms=frame_info.timestamp_ms,
            )
            result.total_frames_processed += 1
            result.total_detections += len(det_frame.detections)

            active_tracks = self.tracker.update(det_frame)
            frame_alerts = []

            for track in active_tracks:
                # find the matching detection to get the crop
                matching_det = next(
                    (d for d in det_frame.detections if d.track_id == track.track_id),
                    None,
                )

                if matching_det is None:
                    # track exists but no detection this frame — use cached attributes
                    cached = self.classifier.get_cached(track.track_id)
                    if cached:
                        track.last_attributes = self.classifier.flatten_attributes(cached)
                    continue

                if self.tracker.should_classify(track, frame_info.frame_idx):
                    crop = self.detector.crop(frame_info.frame, matching_det)
                    pred = self.classifier.classify_crop(
                        crop, track.road_user_type, track.track_id
                    )
                    track.last_attributes = self.classifier.flatten_attributes(pred)
                    result.total_classifications += 1

                if track.last_attributes:
                    alerts = self.rule_engine.evaluate(
                        track_id=track.track_id,
                        road_user_type=track.road_user_type,
                        attributes=track.last_attributes,
                        frame_idx=frame_info.frame_idx,
                        timestamp_ms=frame_info.timestamp_ms,
                        bbox=track.bbox,
                    )
                    for a in alerts:
                        result.alerts.append(a.to_dict())
                        frame_alerts.append(a.to_dict())
                        if on_alert:
                            on_alert(a)

                # update track summary
                result.track_summaries[track.track_id] = {
                    "track_id": track.track_id,
                    "road_user_type": track.road_user_type,
                    "age_frames": track.age,
                    "last_attributes": track.last_attributes,
                    "last_bbox": list(track.bbox),
                }

            self.rule_engine.clear_stale_cooldowns(frame_info.frame_idx)

            yield {
                "frame_idx": frame_info.frame_idx,
                "timestamp_ms": frame_info.timestamp_ms,
                "detections": len(det_frame.detections),
                "active_tracks": len(active_tracks),
                "alerts": frame_alerts,
            }
