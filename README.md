# Multimodal Perception Attribute Classifier

A vision-based classification system that identifies and labels attributes of road users (vehicles, pedestrians, cyclists) from images. Includes a VLM-assisted auto-labeling pipeline with ontology mining to reduce manual annotation effort.

## Overview

- Fine-tunes a Vision Transformer (ViT) on road user images to classify behavioral and physical attributes
- Uses BLIP-2 or LLaVA to auto-generate labels for unlabeled images via ontology-guided prompting
- Serves predictions through a FastAPI REST API
- Tracks experiments and model versions with MLflow
- Stores datasets and artifacts on AWS S3

## Attribute Categories

Eight behavioral and physical attribute categories are supported:

1. **Mobility** - moving, stationary, slow-moving
2. **Orientation** - facing toward, away, lateral
3. **Occlusion** - none, partial, heavy
4. **Lighting** - well-lit, low-light, backlit
5. **Size** - small, medium, large (relative to frame)
6. **Posture** - upright, leaning, crouched (pedestrians/cyclists)
7. **Group** - solo, pair, group
8. **Attention** - attentive, distracted, phone-use (pedestrians)

## Project Structure

```
multimodal-perception-classifier/
├── configs/
│   ├── model_config.yaml         # ViT training hyperparameters
│   └── pipeline_config.yaml      # VLM pipeline and S3 settings
├── src/
│   ├── dataset/
│   │   ├── loader.py             # Dataset loading from BDD100K/nuScenes/COCO
│   │   ├── augmentations.py      # Image augmentation pipeline
│   │   └── ontology.py           # Attribute ontology definitions
│   ├── models/
│   │   ├── vit_classifier.py     # ViT fine-tuning wrapper
│   │   └── attribute_head.py     # Multi-label classification head
│   ├── pipeline/
│   │   ├── vlm_labeler.py        # BLIP-2/LLaVA auto-labeling
│   │   ├── ontology_miner.py     # Ontology-guided prompt construction
│   │   └── label_validator.py    # Label quality filtering
│   ├── serving/
│   │   ├── api.py                # FastAPI application
│   │   ├── inference.py          # Model inference logic
│   │   └── schemas.py            # Pydantic request/response schemas
│   └── tracking/
│       ├── mlflow_logger.py      # MLflow experiment tracking
│       └── s3_handler.py         # AWS S3 artifact management
├── scripts/
│   ├── train.py                  # Training entry point
│   ├── evaluate.py               # Evaluation and metrics
│   ├── run_auto_labeling.py      # VLM labeling pipeline
│   └── upload_artifacts.py       # Push model/data to S3
├── tests/
│   ├── test_dataset.py
│   ├── test_model.py
│   ├── test_api.py
│   └── test_pipeline.py
├── notebooks/
│   └── exploration.ipynb         # Data exploration notebook
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
└── .env.example
```

### Dataset

Download one of the supported datasets:

- **BDD100K** (recommended): https://bdd-data.berkeley.edu/
- **nuScenes**: https://www.nuscenes.org/
- **COCO**: https://cocodataset.org/

## Results

| Model    | Dataset       | Accuracy | F1 (macro) |
| -------- | ------------- | -------- | ---------- |
| ViT-B/16 | BDD100K (50K) | 91.2%    | 89.4%      |
| ViT-L/16 | BDD100K (50K) | 92.8%    | 91.1%      |

Auto-labeling pipeline reduces manual annotation time by ~60% across 8 attribute categories.
