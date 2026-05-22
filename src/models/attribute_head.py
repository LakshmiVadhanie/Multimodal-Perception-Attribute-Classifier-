"""
Multi-task classification head for road user attributes.

Each attribute is a separate classification head sharing a common
backbone representation.
"""

from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.dataset.ontology import ONTOLOGY, get_num_classes


class AttributeHead(nn.Module):
    """
    Single attribute classification head.
    Maps a feature vector to class logits for one attribute.
    """

    def __init__(self, in_features: int, num_classes: int, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.fc = nn.Linear(in_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self.dropout(x))


class MultiAttributeHead(nn.Module):
    """
    Multi-task classification head with one head per attribute.

    Takes the [CLS] token representation from ViT and produces
    independent logits for each attribute category.
    """

    def __init__(
        self,
        in_features: int,
        attribute_names: Optional[List[str]] = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.attribute_names = attribute_names or list(ONTOLOGY.keys())
        self.heads = nn.ModuleDict({
            name: AttributeHead(in_features, get_num_classes(name), dropout)
            for name in self.attribute_names
        })

    def forward(self, features: torch.Tensor) -> Dict[str, torch.Tensor]:
        return {name: head(features) for name, head in self.heads.items()}


class AttributeLoss(nn.Module):
    """
    Computes cross-entropy loss for each attribute independently.

    Handles -1 labels (not applicable or unlabeled) by masking them out.
    Optionally applies per-class weights for imbalanced datasets.
    """

    def __init__(
        self,
        attribute_names: Optional[List[str]] = None,
        attribute_weights: Optional[Dict[str, float]] = None,
    ):
        super().__init__()
        self.attribute_names = attribute_names or list(ONTOLOGY.keys())
        # weight each attribute's loss contribution equally by default
        self.attribute_weights = attribute_weights or {
            name: 1.0 for name in self.attribute_names
        }

    def forward(
        self,
        logits: Dict[str, torch.Tensor],
        labels: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        losses = {}
        for name in self.attribute_names:
            if name not in logits or name not in labels:
                continue

            attr_logits = logits[name]   # (B, num_classes)
            attr_labels = labels[name]   # (B,)

            # mask out -1 labels (not applicable / unlabeled)
            valid_mask = attr_labels >= 0
            if valid_mask.sum() == 0:
                continue

            loss = F.cross_entropy(
                attr_logits[valid_mask],
                attr_labels[valid_mask],
                reduction="mean",
            )
            losses[name] = loss * self.attribute_weights[name]

        if len(losses) == 0:
            return {"total": torch.tensor(0.0, requires_grad=True)}

        total = sum(losses.values())
        losses["total"] = total
        return losses
