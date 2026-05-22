"""
Vision Transformer fine-tuning wrapper for road user attribute classification.

Loads a pretrained ViT from Hugging Face, optionally freezes early layers,
and attaches a multi-task attribute head on top of the [CLS] representation.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from loguru import logger
from transformers import ViTModel, ViTConfig

from src.models.attribute_head import AttributeLoss, MultiAttributeHead
from src.dataset.ontology import ONTOLOGY


class ViTAttributeClassifier(nn.Module):
    """
    Fine-tuned ViT for multi-attribute road user classification.

    Architecture:
        ViT backbone (pretrained) -> [CLS] token -> MultiAttributeHead
    """

    def __init__(
        self,
        model_name: str = "google/vit-base-patch16-224",
        attribute_names: Optional[List[str]] = None,
        dropout: float = 0.1,
        freeze_backbone_layers: int = 0,
    ):
        super().__init__()
        self.attribute_names = attribute_names or list(ONTOLOGY.keys())

        logger.info(f"Loading ViT backbone: {model_name}")
        self.backbone = ViTModel.from_pretrained(model_name, add_pooling_layer=False)
        hidden_size = self.backbone.config.hidden_size

        self.classifier = MultiAttributeHead(
            in_features=hidden_size,
            attribute_names=self.attribute_names,
            dropout=dropout,
        )

        if freeze_backbone_layers > 0:
            self._freeze_layers(freeze_backbone_layers)

    def _freeze_layers(self, num_layers: int):
        # freeze embeddings
        for param in self.backbone.embeddings.parameters():
            param.requires_grad = False

        # freeze the first num_layers transformer blocks
        for i, layer in enumerate(self.backbone.encoder.layer):
            if i < num_layers:
                for param in layer.parameters():
                    param.requires_grad = False

        frozen_params = sum(p.numel() for p in self.parameters() if not p.requires_grad)
        total_params = sum(p.numel() for p in self.parameters())
        logger.info(
            f"Froze {num_layers} backbone layers. "
            f"Trainable params: {total_params - frozen_params:,} / {total_params:,}"
        )

    def forward(self, pixel_values: torch.Tensor) -> Dict[str, torch.Tensor]:
        outputs = self.backbone(pixel_values=pixel_values)
        cls_features = outputs.last_hidden_state[:, 0, :]  # [CLS] token
        logits = self.classifier(cls_features)
        return logits

    def predict(self, pixel_values: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Returns class probabilities (softmax over logits) per attribute."""
        with torch.no_grad():
            logits = self.forward(pixel_values)
        return {name: torch.softmax(l, dim=-1) for name, l in logits.items()}

    def save(self, path: str):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), path / "model.pt")
        # also save the backbone config so we can reconstruct without HF
        self.backbone.config.save_pretrained(path)
        logger.info(f"Model saved to {path}")

    @classmethod
    def load(
        cls,
        path: str,
        model_name: str = "google/vit-base-patch16-224",
        attribute_names: Optional[List[str]] = None,
        device: str = "cpu",
    ) -> "ViTAttributeClassifier":
        model = cls(
            model_name=model_name,
            attribute_names=attribute_names,
        )
        state_dict = torch.load(Path(path) / "model.pt", map_location=device)
        model.load_state_dict(state_dict)
        model.eval()
        logger.info(f"Model loaded from {path}")
        return model


def build_optimizer(model: ViTAttributeClassifier, lr: float, weight_decay: float):
    """
    Different learning rates for backbone vs. classification head.
    The head is trained from scratch so it needs a higher lr.
    """
    backbone_params = [
        p for n, p in model.named_parameters()
        if "backbone" in n and p.requires_grad
    ]
    head_params = [
        p for n, p in model.named_parameters()
        if "classifier" in n and p.requires_grad
    ]

    return torch.optim.AdamW([
        {"params": backbone_params, "lr": lr},
        {"params": head_params, "lr": lr * 10},
    ], weight_decay=weight_decay)


def build_scheduler(optimizer, num_warmup_steps: int, num_training_steps: int):
    from torch.optim.lr_scheduler import LambdaLR

    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        return max(0.0, 0.5 * (1.0 + torch.cos(torch.tensor(progress * 3.14159)).item()))

    return LambdaLR(optimizer, lr_lambda)
