"""Image transforms for distillation."""

from __future__ import annotations

from typing import Tuple

from torch import Tensor
from torchvision import transforms as T


class DistillationTransform:
    """
    Strong augmentation pipeline for distillation.
    
    Same augmentations for teacher and student inputs.
    
    Args:
        image_size: Target image size.
        mean: Normalization mean.
        std: Normalization std.
        min_scale: Minimum scale for random crop.
        max_scale: Maximum scale for random crop.
    """
    
    def __init__(
        self,
        image_size: int | Tuple[int, int] = 224,
        mean: Tuple[float, ...] = (0.485, 0.456, 0.406),
        std: Tuple[float, ...] = (0.229, 0.224, 0.225),
        min_scale: float = 0.4,
        max_scale: float = 1.0,
    ) -> None:
        if isinstance(image_size, int):
            image_size = (image_size, image_size)
        
        self.transform = T.Compose([
            T.RandomResizedCrop(
                size=image_size,
                scale=(min_scale, max_scale),
                interpolation=T.InterpolationMode.BICUBIC,
            ),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomApply([
                T.ColorJitter(
                    brightness=0.4,
                    contrast=0.4,
                    saturation=0.2,
                    hue=0.1,
                )
            ], p=0.8),
            T.RandomGrayscale(p=0.2),
            T.GaussianBlur(kernel_size=23, sigma=(0.1, 2.0)),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ])
    
    def __call__(self, image) -> Tensor:
        return self.transform(image)


class SimpleTransform:
    """
    Simple transform for validation.
    
    Args:
        image_size: Target image size.
        mean: Normalization mean.
        std: Normalization std.
    """
    
    def __init__(
        self,
        image_size: int | Tuple[int, int] = 224,
        mean: Tuple[float, ...] = (0.485, 0.456, 0.406),
        std: Tuple[float, ...] = (0.229, 0.224, 0.225),
    ) -> None:
        if isinstance(image_size, int):
            image_size = (image_size, image_size)
        
        self.transform = T.Compose([
            T.Resize(size=int(image_size[0] * 256 / 224)),
            T.CenterCrop(size=image_size),
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ])
    
    def __call__(self, image) -> Tensor:
        return self.transform(image)
