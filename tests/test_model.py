"""Tests for model architecture and attribute head."""

import pytest
import torch
import torch.nn as nn

from src.models.attribute_head import AttributeHead, MultiAttributeHead, AttributeLoss
from src.dataset.ontology import ONTOLOGY, get_num_classes


class TestAttributeHead:
    def test_output_shape(self):
        head = AttributeHead(in_features=768, num_classes=3, dropout=0.0)
        x = torch.randn(4, 768)
        out = head(x)
        assert out.shape == (4, 3)

    def test_dropout_applied_in_train_mode(self):
        head = AttributeHead(in_features=768, num_classes=3, dropout=0.9)
        head.train()
        x = torch.ones(100, 768)
        out1 = head(x)
        out2 = head(x)
        # with high dropout, outputs should differ in training mode
        assert not torch.allclose(out1, out2)

    def test_no_dropout_in_eval_mode(self):
        head = AttributeHead(in_features=768, num_classes=3, dropout=0.9)
        head.eval()
        x = torch.ones(10, 768)
        out1 = head(x)
        out2 = head(x)
        assert torch.allclose(out1, out2)


class TestMultiAttributeHead:
    def test_output_keys(self):
        head = MultiAttributeHead(in_features=768)
        x = torch.randn(4, 768)
        out = head(x)
        assert set(out.keys()) == set(ONTOLOGY.keys())

    def test_output_shapes(self):
        head = MultiAttributeHead(in_features=768)
        x = torch.randn(4, 768)
        out = head(x)
        for attr_name, logits in out.items():
            expected_classes = get_num_classes(attr_name)
            assert logits.shape == (4, expected_classes), (
                f"{attr_name}: expected (4, {expected_classes}), got {logits.shape}"
            )

    def test_custom_attribute_subset(self):
        subset = ["mobility", "occlusion", "lighting"]
        head = MultiAttributeHead(in_features=512, attribute_names=subset)
        x = torch.randn(2, 512)
        out = head(x)
        assert set(out.keys()) == set(subset)


class TestAttributeLoss:
    def _make_batch(self, batch_size=4):
        logits = {
            name: torch.randn(batch_size, get_num_classes(name))
            for name in ONTOLOGY
        }
        labels = {
            name: torch.randint(0, get_num_classes(name), (batch_size,))
            for name in ONTOLOGY
        }
        return logits, labels

    def test_loss_is_positive(self):
        loss_fn = AttributeLoss()
        logits, labels = self._make_batch()
        losses = loss_fn(logits, labels)
        assert losses["total"].item() > 0

    def test_masked_labels_excluded(self):
        loss_fn = AttributeLoss()
        logits, labels = self._make_batch(batch_size=4)
        # set all labels for "attention" to -1 (not applicable)
        labels["attention"] = torch.full((4,), -1, dtype=torch.long)
        losses = loss_fn(logits, labels)
        assert "attention" not in losses
        assert losses["total"].item() > 0

    def test_all_masked_returns_zero(self):
        loss_fn = AttributeLoss(attribute_names=["mobility"])
        logits = {"mobility": torch.randn(4, 3)}
        labels = {"mobility": torch.full((4,), -1, dtype=torch.long)}
        losses = loss_fn(logits, labels)
        # no valid samples -> total should be 0
        assert losses["total"].item() == pytest.approx(0.0, abs=1e-6)

    def test_gradients_flow(self):
        loss_fn = AttributeLoss()
        head = MultiAttributeHead(in_features=64)
        x = torch.randn(4, 64, requires_grad=True)
        logits = head(x)
        labels = {
            name: torch.randint(0, get_num_classes(name), (4,))
            for name in ONTOLOGY
        }
        losses = loss_fn(logits, labels)
        losses["total"].backward()
        assert x.grad is not None
