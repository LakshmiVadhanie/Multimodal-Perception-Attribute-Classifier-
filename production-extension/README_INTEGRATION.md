# Integration Guide

This folder contains the 6 new modules that upgrade the classifier
into a full production pipeline. Here is exactly how to plug them
into the original project.

## File layout after merging

Drop these files directly into the root of `multimodal-perception-classifier/`:

```
multimodal-perception-classifier/
├── detection/
│   └── detector.py          <-- NEW
├── tracking/
│   └── tracker.py           <-- NEW
├── classification/
│   └── classifier_bridge.py <-- NEW
├── ingestion/
│   └── video_ingestor.py    <-- NEW
├── alerts/
│   └── rule_engine.py       <-- NEW
├── serving/
│   ├── pipeline.py          <-- NEW (orchestrator)
│   └── api_v2.py            <-- NEW (replaces src/serving/api.py)
├── dashboard/
│   └── app.py               <-- NEW
├── src/                     <- original unchanged
├── configs/                 <- original unchanged
├── scripts/                 <- original unchanged
└── ...
```

## Step 1 — install new dependencies

```bash
pip install ultralytics opencv-python streamlit
```

The rest (fastapi, torch, transformers, etc.) are already installed
from the original requirements.txt.

## Step 2 — train or download the ViT classifier

The pipeline expects a trained checkpoint at `./checkpoints/best_model/`.

If you do not have one yet, run the original training script:
```bash
python scripts/train.py --config configs/model_config.yaml
```

Or, to test the pipeline without training, you can skip classification
by commenting out `self.classifier.load()` in `serving/pipeline.py`
and having `classify_crop` return empty attributes. Detection and
alerting will still work.

## Step 3 — verify YOLO weights download

On first run, YOLOv8 will automatically download `yolov8n.pt` (~6MB)
from Ultralytics servers. If you are offline, download it manually:

```bash
pip install ultralytics
python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"
```

The weights land in `~/.config/Ultralytics/` and are reused automatically.

## Step 4 — start the API

```bash
# from the multimodal-perception-classifier/ root
uvicorn serving.api_v2:app --host 0.0.0.0 --port 8000 --reload
```

Environment variables you can set:
```bash
export MODEL_PATH=./checkpoints/best_model
export MODEL_NAME=google/vit-base-patch16-224
export YOLO_WEIGHTS=yolov8n.pt   # or yolov8m.pt for better accuracy
export DEVICE=mps                # mps for Apple Silicon, cuda, or cpu
```

## Step 5 — start the dashboard

```bash
streamlit run dashboard/app.py
```

Open http://localhost:8501 in your browser.

## Step 6 — test with an image

```bash
curl -X POST http://localhost:8000/analyze/image \
  -F "file=@/path/to/street_photo.jpg"
```

## How the pieces connect

```
User uploads image/video
        |
        v
api_v2.py  (FastAPI)
        |
        v
pipeline.py  (orchestrator)
        |
        +---> video_ingestor.py  (reads frames from file or RTSP)
        |
        +---> detector.py        (YOLOv8 → bounding boxes per frame)
        |
        +---> tracker.py         (assigns consistent IDs across frames)
        |
        +---> classifier_bridge.py  (crops → ViT → attribute labels)
        |
        +---> rule_engine.py     (labels → alert conditions → Alert objects)
        |
        v
JSON response  /  SSE stream  /  job result
        |
        v
dashboard/app.py  (Streamlit UI displays results + alert log)
```

## Adding a new alert rule

No code changes needed. Edit `alerts/rule_engine.py`, find `DEFAULT_RULES`,
and add a new entry:

```python
{
    "id": "my_new_rule",
    "name": "Description shown in dashboard",
    "severity": "high",          # low | medium | high | critical
    "conditions": {
        "road_user_type": "pedestrian",   # or "vehicle", "cyclist", "all"
        "attributes": {
            "attention": "distracted",
            "lighting": "low_light",
        },
    },
},
```

The engine checks all conditions with AND logic — all specified
attributes must match for the alert to fire.

## Swapping in a better tracker

The `IoUTracker` in `tracking/tracker.py` is intentionally simple.
To use ByteTrack (much better for fast-moving cameras):

```bash
pip install ultralytics  # ByteTrack is bundled
```

In `serving/pipeline.py`, replace the tracker instantiation:

```python
# instead of:
self.tracker = IoUTracker(...)

# use:
from ultralytics.trackers import BYTETracker
self.tracker = BYTETracker(args)  # see ultralytics docs for args
```

The rest of the pipeline does not need to change because `BYTETracker`
returns the same bbox + track_id structure.

## Mac-specific notes

- Set `DEVICE=mps` for Apple Silicon. Both YOLOv8 and the ViT classifier
  support MPS acceleration.
- Processing a 1080p video at 10fps (process_every_n_frames=3 on 30fps input)
  takes roughly 2-4x realtime on an M2 without a trained classifier.
  With the classifier it will be slower — reduce to process_every_n_frames=6
  if it feels sluggish.
- RTSP streams work on Mac via the system FFmpeg that ships with OpenCV.
  If you hit issues, install `ffmpeg` via Homebrew: `brew install ffmpeg`.
