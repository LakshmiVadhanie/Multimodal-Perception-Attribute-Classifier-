"""
tracking/tracker.py

Lightweight IoU-based object tracker.

Assigns consistent track IDs to detections across frames so we do not
re-classify the same road user on every frame. No external dependency —
pure numpy.

For a production upgrade, swap this out with ByteTrack or BoT-SORT
(both available via ultralytics). The interface is identical.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from detection.detector import Detection, DetectionFrame


@dataclass
class Track:
    track_id: int
    road_user_type: str
    bbox: Tuple[int, int, int, int]
    last_seen_frame: int
    age: int = 1                    # frames this track has been alive
    missed_frames: int = 0          # consecutive frames with no matching detection
    last_attributes: Optional[Dict] = None  # cached classifier output


class IoUTracker:
    """
    Greedy IoU matching tracker.

    Each frame, existing tracks are matched to new detections by highest
    IoU. Unmatched detections start new tracks. Tracks that go unmatched
    for more than `max_missed_frames` are dropped.

    This is intentionally simple. It works well for fixed-camera scenes
    where road users move slowly between frames. For fast dashcam footage
    use ByteTrack instead.
    """

    def __init__(
        self,
        iou_threshold: float = 0.35,
        max_missed_frames: int = 10,
        min_age_to_classify: int = 2,   # skip classification on brand-new tracks
        classify_every_n_frames: int = 5,  # only re-classify every N frames
    ):
        self.iou_threshold = iou_threshold
        self.max_missed_frames = max_missed_frames
        self.min_age_to_classify = min_age_to_classify
        self.classify_every_n_frames = classify_every_n_frames

        self._tracks: Dict[int, Track] = {}
        self._next_id = 1

    def update(self, detection_frame: DetectionFrame) -> List[Track]:
        """
        Match detections to existing tracks, update state, return active tracks.
        Each returned track has its detection's bbox attached.
        """
        detections = detection_frame.detections
        frame_idx = detection_frame.frame_idx

        if not detections:
            self._age_tracks(frame_idx)
            return list(self._tracks.values())

        if not self._tracks:
            for det in detections:
                self._new_track(det, frame_idx)
            return list(self._tracks.values())

        track_ids = list(self._tracks.keys())
        track_bboxes = np.array([self._tracks[tid].bbox for tid in track_ids])
        det_bboxes = np.array([d.bbox for d in detections])

        iou_matrix = self._batch_iou(track_bboxes, det_bboxes)

        matched_tracks = set()
        matched_dets = set()

        # greedy: always match the highest-IoU pair first
        while True:
            if iou_matrix.size == 0:
                break
            max_iou = iou_matrix.max()
            if max_iou < self.iou_threshold:
                break
            t_idx, d_idx = np.unravel_index(iou_matrix.argmax(), iou_matrix.shape)
            tid = track_ids[t_idx]

            track = self._tracks[tid]
            det = detections[d_idx]

            # only match if same road user type
            if track.road_user_type == det.road_user_type:
                track.bbox = det.bbox
                track.last_seen_frame = frame_idx
                track.age += 1
                track.missed_frames = 0
                det.track_id = tid
                matched_tracks.add(t_idx)
                matched_dets.add(d_idx)

            iou_matrix[t_idx, :] = -1
            iou_matrix[:, d_idx] = -1

        # unmatched detections become new tracks
        for d_idx, det in enumerate(detections):
            if d_idx not in matched_dets:
                self._new_track(det, frame_idx)

        # age unmatched tracks
        for t_idx, tid in enumerate(track_ids):
            if t_idx not in matched_tracks:
                self._tracks[tid].missed_frames += 1

        # drop stale tracks
        stale = [
            tid for tid, t in self._tracks.items()
            if t.missed_frames > self.max_missed_frames
        ]
        for tid in stale:
            del self._tracks[tid]

        return list(self._tracks.values())

    def should_classify(self, track: Track, frame_idx: int) -> bool:
        """True if this track should be sent to the attribute classifier this frame."""
        if track.age < self.min_age_to_classify:
            return False
        if track.last_attributes is None:
            return True
        return frame_idx % self.classify_every_n_frames == 0

    def _new_track(self, det: Detection, frame_idx: int):
        track = Track(
            track_id=self._next_id,
            road_user_type=det.road_user_type,
            bbox=det.bbox,
            last_seen_frame=frame_idx,
        )
        det.track_id = self._next_id
        self._tracks[self._next_id] = track
        self._next_id += 1

    def _age_tracks(self, frame_idx: int):
        for track in self._tracks.values():
            track.missed_frames += 1
        stale = [
            tid for tid, t in self._tracks.items()
            if t.missed_frames > self.max_missed_frames
        ]
        for tid in stale:
            del self._tracks[tid]

    @staticmethod
    def _batch_iou(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
        """Compute IoU matrix between two sets of boxes [N,4] and [M,4]."""
        ax1, ay1, ax2, ay2 = boxes_a[:, 0], boxes_a[:, 1], boxes_a[:, 2], boxes_a[:, 3]
        bx1, by1, bx2, by2 = boxes_b[:, 0], boxes_b[:, 1], boxes_b[:, 2], boxes_b[:, 3]

        inter_x1 = np.maximum(ax1[:, None], bx1[None, :])
        inter_y1 = np.maximum(ay1[:, None], by1[None, :])
        inter_x2 = np.minimum(ax2[:, None], bx2[None, :])
        inter_y2 = np.minimum(ay2[:, None], by2[None, :])

        inter_w = np.maximum(0, inter_x2 - inter_x1)
        inter_h = np.maximum(0, inter_y2 - inter_y1)
        intersection = inter_w * inter_h

        area_a = (ax2 - ax1) * (ay2 - ay1)
        area_b = (bx2 - bx1) * (by2 - by1)
        union = area_a[:, None] + area_b[None, :] - intersection

        return np.where(union > 0, intersection / union, 0.0)
