"""
ingestion/video_ingestor.py

Handles frame extraction from:
  - Local video files (.mp4, .avi, .mov, etc.)
  - RTSP camera streams (rtsp://...)
  - Single image uploads

Yields frames as numpy arrays at a controlled framerate so the
downstream pipeline is not overwhelmed.
"""

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Generator, Optional, Tuple, Union

import cv2
import numpy as np
from loguru import logger


@dataclass
class FrameInfo:
    frame: np.ndarray          # BGR numpy array
    frame_idx: int
    timestamp_ms: float
    source: str                # file path or stream URL


class VideoIngestor:
    """
    Reads frames from a video file or RTSP stream.

    process_every_n_frames controls how many source frames to skip.
    For a 30fps camera with process_every_n_frames=3, you analyze 10fps.
    This is usually plenty for attribute classification.
    """

    def __init__(
        self,
        process_every_n_frames: int = 3,
        max_width: int = 1280,     # resize wide frames to keep inference fast
        reconnect_attempts: int = 5,
        reconnect_delay_s: float = 2.0,
    ):
        self.process_every_n_frames = process_every_n_frames
        self.max_width = max_width
        self.reconnect_attempts = reconnect_attempts
        self.reconnect_delay_s = reconnect_delay_s

    def stream_frames(self, source: Union[str, Path]) -> Generator[FrameInfo, None, None]:
        """
        Yields FrameInfo objects from the source.
        Works for video files and RTSP URLs.
        """
        source = str(source)
        is_stream = source.startswith("rtsp://") or source.startswith("rtmp://")

        if is_stream:
            yield from self._stream_rtsp(source)
        else:
            yield from self._stream_file(source)

    def _stream_file(self, path: str) -> Generator[FrameInfo, None, None]:
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            raise FileNotFoundError(f"Cannot open video: {path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        logger.info(f"Video: {path} | {total} frames @ {fps:.1f}fps")

        frame_idx = 0
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_idx % self.process_every_n_frames == 0:
                    frame = self._resize(frame)
                    timestamp_ms = (frame_idx / fps) * 1000
                    yield FrameInfo(
                        frame=frame,
                        frame_idx=frame_idx,
                        timestamp_ms=timestamp_ms,
                        source=path,
                    )

                frame_idx += 1
        finally:
            cap.release()

    def _stream_rtsp(self, url: str) -> Generator[FrameInfo, None, None]:
        frame_idx = 0
        attempts = 0

        while attempts <= self.reconnect_attempts:
            cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # minimize latency

            if not cap.isOpened():
                attempts += 1
                logger.warning(f"RTSP connection failed (attempt {attempts}). Retrying...")
                time.sleep(self.reconnect_delay_s)
                continue

            logger.info(f"RTSP stream connected: {url}")
            attempts = 0  # reset on successful connection

            try:
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        logger.warning("RTSP stream lost. Reconnecting...")
                        break

                    if frame_idx % self.process_every_n_frames == 0:
                        frame = self._resize(frame)
                        timestamp_ms = frame_idx * (1000.0 / 30.0)  # assume 30fps for streams
                        yield FrameInfo(
                            frame=frame,
                            frame_idx=frame_idx,
                            timestamp_ms=timestamp_ms,
                            source=url,
                        )

                    frame_idx += 1
            finally:
                cap.release()

            attempts += 1
            time.sleep(self.reconnect_delay_s)

        raise ConnectionError(f"RTSP stream unavailable after {self.reconnect_attempts} attempts: {url}")

    def _resize(self, frame: np.ndarray) -> np.ndarray:
        h, w = frame.shape[:2]
        if w <= self.max_width:
            return frame
        scale = self.max_width / w
        new_w = int(w * scale)
        new_h = int(h * scale)
        return cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    @staticmethod
    def load_image(path: Union[str, Path]) -> FrameInfo:
        """Load a single image file as a FrameInfo for one-shot analysis."""
        frame = cv2.imread(str(path))
        if frame is None:
            raise FileNotFoundError(f"Cannot read image: {path}")
        return FrameInfo(
            frame=frame,
            frame_idx=0,
            timestamp_ms=0.0,
            source=str(path),
        )
