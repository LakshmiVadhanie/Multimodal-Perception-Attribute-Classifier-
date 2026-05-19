"""
serving/api_v2.py

Production FastAPI replacing the original single-image /predict endpoint.

Endpoints:
    POST /analyze/image   - upload an image, get back all detections + attributes + alerts
    POST /analyze/video   - upload a video file, process async, return job ID
    GET  /jobs/{job_id}   - poll job status and results
    GET  /stream/alerts   - SSE endpoint for live alert push from a running stream
    GET  /alerts          - query the alert log with filters
    GET  /health          - liveness check

Run:
    uvicorn serving.api_v2:app --host 0.0.0.0 --port 8000
"""

import asyncio
import io
import json
import os
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncGenerator, Dict, Optional

from fastapi import BackgroundTasks, FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from loguru import logger
from PIL import Image

from serving.pipeline import VideoPipeline


MODEL_PATH = os.getenv("MODEL_PATH", "./checkpoints/best_model")
MODEL_NAME = os.getenv("MODEL_NAME", "google/vit-base-patch16-224")
YOLO_WEIGHTS = os.getenv("YOLO_WEIGHTS", "yolov8n.pt")
DEVICE = os.getenv("DEVICE", None)
UPLOAD_DIR = Path(os.getenv("UPLOAD_DIR", "/tmp/pipeline_uploads"))
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

pipeline = VideoPipeline(
    model_path=MODEL_PATH,
    model_name=MODEL_NAME,
    yolo_weights=YOLO_WEIGHTS,
    device=DEVICE,
)

# in-memory job store (swap for Redis in multi-worker deployments)
jobs: Dict[str, Dict] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    pipeline.load()
    yield


app = FastAPI(
    title="Road User Attribute Analysis API",
    description="Detects and classifies road users in images and video. Returns attribute labels and safety alerts.",
    version="2.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": True, "device": pipeline.classifier._inference.device}


@app.post("/analyze/image")
async def analyze_image(
    file: UploadFile = File(..., description="JPEG, PNG, or WEBP image"),
):
    """
    Upload a single image. Returns all detected road users with
    attribute labels and any triggered alerts.
    """
    if file.content_type not in ("image/jpeg", "image/png", "image/webp"):
        raise HTTPException(status_code=415, detail="Unsupported file type")

    contents = await file.read()
    suffix = Path(file.filename or "upload.jpg").suffix
    tmp_path = UPLOAD_DIR / f"{uuid.uuid4()}{suffix}"

    try:
        tmp_path.write_bytes(contents)
        result = pipeline.run_on_image(tmp_path)
    except Exception as e:
        logger.exception("Image analysis failed")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        tmp_path.unlink(missing_ok=True)

    return result


@app.post("/analyze/video")
async def analyze_video(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(..., description="MP4, AVI, or MOV video file"),
):
    """
    Upload a video file. Processing runs in the background.
    Returns a job_id — poll /jobs/{job_id} for status and results.
    """
    job_id = str(uuid.uuid4())
    suffix = Path(file.filename or "upload.mp4").suffix
    video_path = UPLOAD_DIR / f"{job_id}{suffix}"
    result_path = UPLOAD_DIR / f"{job_id}_result.json"

    contents = await file.read()
    video_path.write_bytes(contents)

    jobs[job_id] = {"status": "queued", "job_id": job_id, "result": None}
    background_tasks.add_task(_process_video_job, job_id, video_path, result_path)

    return {"job_id": job_id, "status": "queued"}


@app.get("/jobs/{job_id}")
def get_job(job_id: str):
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job


@app.get("/alerts")
def get_alerts(
    severity: Optional[str] = Query(None, description="low | medium | high | critical"),
    road_user_type: Optional[str] = Query(None, description="vehicle | pedestrian | cyclist"),
    limit: int = Query(100, ge=1, le=1000),
):
    """Return the accumulated alert log with optional filters."""
    alerts = pipeline.rule_engine.get_alerts(
        severity_filter=severity,
        road_user_filter=road_user_type,
    )
    return {"alerts": alerts[-limit:], "total": len(alerts)}


@app.get("/stream/alerts")
async def stream_alerts(source: str = Query(..., description="RTSP URL or video file path")):
    """
    Server-Sent Events endpoint.
    Connect from the dashboard to receive live alert pushes as the pipeline
    processes a stream.

    EventSource usage (browser):
        const es = new EventSource('/stream/alerts?source=rtsp://...')
        es.onmessage = e => console.log(JSON.parse(e.data))
    """
    async def event_generator() -> AsyncGenerator[str, None]:
        loop = asyncio.get_event_loop()
        try:
            for frame_result in pipeline.stream_results(source):
                if frame_result["alerts"]:
                    data = json.dumps(frame_result)
                    yield f"data: {data}\n\n"
                await asyncio.sleep(0)  # yield control to event loop
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


def _process_video_job(job_id: str, video_path: Path, result_path: Path):
    jobs[job_id]["status"] = "running"
    try:
        result = pipeline.run_on_file(video_path, output_path=str(result_path))
        jobs[job_id]["status"] = "complete"
        jobs[job_id]["result"] = result.to_dict()
    except Exception as e:
        logger.exception(f"Job {job_id} failed")
        jobs[job_id]["status"] = "failed"
        jobs[job_id]["error"] = str(e)
    finally:
        video_path.unlink(missing_ok=True)
