"""Tests for dataset loading and ontology utilities."""

import pytest
from unittest.mock import MagicMock, patch
from PIL import Image
import numpy as np
import torch

from src.dataset.ontology import (
    ONTOLOGY,
    LABEL_MAPS,
    LABEL_TO_IDX,
    ROAD_USER_TYPES,
    get_applicable_attributes,
    get_num_classes,
)
from src.dataset.augmentations import (
    AugmentationConfig,
    build_train_transforms,
    build_val_transforms,
)


class TestOntology:
    def test_all_attributes_defined(self):
        expected = {"mobility", "orientation", "occlusion", "lighting", "size", "posture", "group", "attention"}
        assert set(ONTOLOGY.keys()) == expected

    def test_label_maps_consistent(self):
        for attr_name, attr in ONTOLOGY.items():
            label_map = LABEL_MAPS[attr_name]
            assert len(label_map) == len(attr.values)
            for i, v in enumerate(attr.values):
                assert label_map[i] == v.label

    def test_label_to_idx_round_trip(self):
        for attr_name in ONTOLOGY:
            label_map = LABEL_MAPS[attr_name]
            idx_map = LABEL_TO_IDX[attr_name]
            for idx, label in label_map.items():
                assert idx_map[label] == idx

    def test_get_applicable_attributes_vehicle(self):
        attrs = get_applicable_attributes("vehicle")
        assert "mobility" in attrs
        assert "orientation" in attrs
        assert "posture" not in attrs  # pedestrian/cyclist only
        assert "attention" not in attrs  # pedestrian only

    def test_get_applicable_attributes_pedestrian(self):
        attrs = get_applicable_attributes("pedestrian")
        assert "posture" in attrs
        assert "attention" in attrs
        assert "mobility" in attrs

    def test_get_applicable_attributes_cyclist(self):
        attrs = get_applicable_attributes("cyclist")
        assert "posture" in attrs
        assert "attention" not in attrs

    def test_get_num_classes(self):
        for attr_name, attr in ONTOLOGY.items():
            assert get_num_classes(attr_name) == len(attr.values)

    def test_all_attributes_have_at_least_two_classes(self):
        for attr_name in ONTOLOGY:
            assert get_num_classes(attr_name) >= 2, f"{attr_name} has fewer than 2 classes"


class TestAugmentations:
    def _make_image(self, size=(224, 224)):
        arr = np.random.randint(0, 255, (*size, 3), dtype=np.uint8)
        return Image.fromarray(arr)

    def test_train_transforms_output_shape(self):
        config = AugmentationConfig(image_size=224)
        transform = build_train_transforms(config)
        img = self._make_image()
        tensor = transform(img)
        assert tensor.shape == (3, 224, 224)
        assert isinstance(tensor, torch.Tensor)

    def test_val_transforms_output_shape(self):
        config = AugmentationConfig(image_size=224)
        transform = build_val_transforms(config)
        img = self._make_image(size=(300, 300))
        tensor = transform(img)
        assert tensor.shape == (3, 224, 224)

    def test_transforms_normalize(self):
        config = AugmentationConfig(image_size=224)
        transform = build_val_transforms(config)
        img = self._make_image()
        tensor = transform(img)
        # normalized tensors should have values outside [0, 1]
        assert tensor.min() < 0 or tensor.max() > 1


class TestRoadUserDataset:
    def _make_annotation(self, image_path="fake.jpg", road_user_type="vehicle"):
        return {
            "image_path": image_path,
            "road_user_type": road_user_type,
            "attributes": {
                "mobility": "moving",
                "occlusion": "none",
                "lighting": "well_lit",
                "orientation": "facing_toward",
                "size": "medium",
                "group": "solo",
            },
        }

    def test_label_encoding(self):
        from src.dataset.ontology import LABEL_TO_IDX
        assert LABEL_TO_IDX["mobility"]["moving"] == 0
        assert LABEL_TO_IDX["mobility"]["stationary"] == 1
        assert LABEL_TO_IDX["mobility"]["slow_moving"] == 2

    def test_not_applicable_attributes_get_minus_one(self):
        # vehicles should not have posture labels
        from src.dataset.ontology import LABEL_TO_IDX, get_applicable_attributes
        applicable = get_applicable_attributes("vehicle")
        assert "posture" not in applicable
        assert "attention" not in applicable
