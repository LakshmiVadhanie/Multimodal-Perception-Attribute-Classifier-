"""
Dataset loading for road user attribute classification.

Supports BDD100K, nuScenes, and COCO-formatted annotation files.
Labels are expected to map image paths to per-attribute class indices.
"""

import json
import os
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from loguru import logger

from src.dataset.ontology import LABEL_TO_IDX, ONTOLOGY, get_applicable_attributes


class RoadUserDataset(Dataset):
    """
    Dataset for road user attribute classification.

    Expects annotations as a list of dicts:
    [
        {
            "image_path": "path/to/image.jpg",
            "road_user_type": "pedestrian",
            "attributes": {
                "mobility": "moving",
                "occlusion": "none",
                ...
            },
            "bbox": [x1, y1, x2, y2]  # optional crop coordinates
        },
        ...
    ]
    """

    def __init__(
        self,
        annotations: List[Dict],
        image_root: str,
        transform: Optional[Callable] = None,
        attribute_names: Optional[List[str]] = None,
    ):
        self.annotations = annotations
        self.image_root = Path(image_root)
        self.transform = transform
        self.attribute_names = attribute_names or list(ONTOLOGY.keys())

        self._validate_annotations()

    def _validate_annotations(self):
        missing = 0
        for ann in self.annotations:
            img_path = self.image_root / ann["image_path"]
            if not img_path.exists():
                missing += 1
        if missing > 0:
            logger.warning(f"{missing}/{len(self.annotations)} image files not found on disk")

    def __len__(self) -> int:
        return len(self.annotations)

    def __getitem__(self, idx: int) -> Dict:
        ann = self.annotations[idx]
        image_path = self.image_root / ann["image_path"]

        image = Image.open(image_path).convert("RGB")

        # crop to bounding box if provided
        if "bbox" in ann and ann["bbox"] is not None:
            x1, y1, x2, y2 = ann["bbox"]
            image = image.crop((x1, y1, x2, y2))

        if self.transform:
            image = self.transform(image)

        road_user_type = ann.get("road_user_type", "vehicle")
        applicable = get_applicable_attributes(road_user_type)

        labels = {}
        for attr_name in self.attribute_names:
            if attr_name in applicable and attr_name in ann.get("attributes", {}):
                label_str = ann["attributes"][attr_name]
                labels[attr_name] = LABEL_TO_IDX[attr_name].get(label_str, -1)
            else:
                labels[attr_name] = -1  # -1 means not applicable or unlabeled

        return {
            "image": image,
            "labels": labels,
            "road_user_type": road_user_type,
            "image_path": str(image_path),
        }


def load_bdd100k_annotations(dataset_path: str) -> List[Dict]:
    """
    Parse BDD100K detection annotations into the unified format.
    BDD100K labels path: <dataset_path>/labels/det_20/det_train.json
    """
    label_path = Path(dataset_path) / "labels" / "det_20" / "det_train.json"
    if not label_path.exists():
        raise FileNotFoundError(f"BDD100K labels not found at {label_path}")

    with open(label_path) as f:
        raw = json.load(f)

    annotations = []
    category_map = {
        "car": "vehicle",
        "truck": "vehicle",
        "bus": "vehicle",
        "motor": "cyclist",
        "bike": "cyclist",
        "person": "pedestrian",
        "rider": "cyclist",
    }

    for item in raw:
        if not item.get("labels"):
            continue
        image_path = f"images/100k/train/{item['name']}"
        for label in item["labels"]:
            category = label.get("category", "").lower()
            if category not in category_map:
                continue
            road_user_type = category_map[category]
            box2d = label.get("box2d")
            bbox = None
            if box2d:
                bbox = [box2d["x1"], box2d["y1"], box2d["x2"], box2d["y2"]]

            annotations.append({
                "image_path": image_path,
                "road_user_type": road_user_type,
                "bbox": bbox,
                "attributes": {},  # attributes filled by auto-labeling or manual
            })

    logger.info(f"Loaded {len(annotations)} BDD100K annotations")
    return annotations


def load_custom_annotations(annotation_file: str) -> List[Dict]:
    """Load from a pre-built JSON annotation file in the unified format."""
    with open(annotation_file) as f:
        data = json.load(f)
    logger.info(f"Loaded {len(data)} annotations from {annotation_file}")
    return data


def split_annotations(
    annotations: List[Dict],
    val_ratio: float = 0.15,
    test_ratio: float = 0.10,
    seed: int = 42,
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(annotations)).tolist()

    n = len(annotations)
    n_test = int(n * test_ratio)
    n_val = int(n * val_ratio)

    test_idx = indices[:n_test]
    val_idx = indices[n_test:n_test + n_val]
    train_idx = indices[n_test + n_val:]

    train = [annotations[i] for i in train_idx]
    val = [annotations[i] for i in val_idx]
    test = [annotations[i] for i in test_idx]

    logger.info(f"Split: {len(train)} train / {len(val)} val / {len(test)} test")
    return train, val, test


def build_weighted_sampler(dataset: RoadUserDataset, attribute_name: str) -> WeightedRandomSampler:
    """
    Build a sampler that up-weights rare classes for a given attribute.
    Useful for handling class imbalance.
    """
    labels = [
        ann["attributes"].get(attribute_name, None)
        for ann in dataset.annotations
    ]
    label_indices = [
        LABEL_TO_IDX[attribute_name].get(l, -1) if l else -1
        for l in labels
    ]

    num_classes = len(ONTOLOGY[attribute_name].values)
    class_counts = np.bincount(
        [l for l in label_indices if l >= 0], minlength=num_classes
    )
    class_weights = 1.0 / np.maximum(class_counts, 1)

    sample_weights = np.array([
        class_weights[l] if l >= 0 else 0.0
        for l in label_indices
    ])

    return WeightedRandomSampler(
        weights=torch.from_numpy(sample_weights).float(),
        num_samples=len(dataset),
        replacement=True,
    )


def build_dataloaders(
    train_dataset: RoadUserDataset,
    val_dataset: RoadUserDataset,
    test_dataset: Optional[RoadUserDataset],
    batch_size: int = 32,
    num_workers: int = 4,
    use_weighted_sampler: bool = False,
) -> Tuple[DataLoader, DataLoader, Optional[DataLoader]]:

    sampler = None
    if use_weighted_sampler:
        sampler = build_weighted_sampler(train_dataset, "mobility")

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = None
    if test_dataset is not None:
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )

    return train_loader, val_loader, test_loader
