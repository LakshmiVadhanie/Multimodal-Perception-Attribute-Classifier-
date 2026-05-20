"""
Image augmentation pipeline for training and inference.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from PIL import Image


@dataclass
class AugmentationConfig:
    image_size: int = 224
    random_horizontal_flip: float = 0.5
    random_crop_scale: Tuple[float, float] = (0.8, 1.0)
    color_jitter_brightness: float = 0.2
    color_jitter_contrast: float = 0.2
    color_jitter_saturation: float = 0.1
    color_jitter_hue: float = 0.05
    normalize_mean: List[float] = None
    normalize_std: List[float] = None

    def __post_init__(self):
        if self.normalize_mean is None:
            self.normalize_mean = [0.485, 0.456, 0.406]
        if self.normalize_std is None:
            self.normalize_std = [0.229, 0.224, 0.225]


def build_train_transforms(config: AugmentationConfig) -> T.Compose:
    return T.Compose([
        T.RandomResizedCrop(
            config.image_size,
            scale=config.random_crop_scale,
            interpolation=T.InterpolationMode.BICUBIC,
        ),
        T.RandomHorizontalFlip(p=config.random_horizontal_flip),
        T.ColorJitter(
            brightness=config.color_jitter_brightness,
            contrast=config.color_jitter_contrast,
            saturation=config.color_jitter_saturation,
            hue=config.color_jitter_hue,
        ),
        T.RandomGrayscale(p=0.05),
        T.ToTensor(),
        T.Normalize(mean=config.normalize_mean, std=config.normalize_std),
    ])


def build_val_transforms(config: AugmentationConfig) -> T.Compose:
    # resize to slightly larger then center crop, standard eval protocol
    resize_size = int(config.image_size * 1.143)
    return T.Compose([
        T.Resize(resize_size, interpolation=T.InterpolationMode.BICUBIC),
        T.CenterCrop(config.image_size),
        T.ToTensor(),
        T.Normalize(mean=config.normalize_mean, std=config.normalize_std),
    ])


def build_inference_transforms(image_size: int = 224) -> T.Compose:
    config = AugmentationConfig(image_size=image_size)
    return build_val_transforms(config)


def denormalize(tensor: torch.Tensor, mean: List[float], std: List[float]) -> torch.Tensor:
    """Reverse normalization for visualization."""
    mean_t = torch.tensor(mean, dtype=tensor.dtype, device=tensor.device).view(3, 1, 1)
    std_t = torch.tensor(std, dtype=tensor.dtype, device=tensor.device).view(3, 1, 1)
    return tensor * std_t + mean_t
