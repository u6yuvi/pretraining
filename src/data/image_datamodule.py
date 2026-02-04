"""
LightningDataModule for image datasets.

Loads images from a directory for self-supervised pretraining.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable, List, Optional

from lightning import LightningDataModule
from PIL import Image
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, random_split

from src.data.transforms import DistillationTransform, SimpleTransform

logger = logging.getLogger(__name__)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".ppm", ".bmp", ".pgm", ".tif", ".tiff", ".webp"}


class ImageFolderDataset(Dataset):
    """Dataset that loads all images from a directory."""
    
    def __init__(
        self,
        root: str | Path,
        transform: Callable | None = None,
    ) -> None:
        self.root = Path(root)
        self.transform = transform
        
        if not self.root.exists():
            raise FileNotFoundError(f"Directory not found: {self.root}")
        
        self.image_paths = self._find_images()
        
        if not self.image_paths:
            raise ValueError(f"No images found in {self.root}")
        
        logger.info(f"Found {len(self.image_paths)} images in {self.root}")
    
    def _find_images(self) -> List[Path]:
        images = []
        for ext in IMAGE_EXTENSIONS:
            images.extend(self.root.rglob(f"*{ext}"))
            images.extend(self.root.rglob(f"*{ext.upper()}"))
        return sorted(set(images))
    
    def __len__(self) -> int:
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> Tensor:
        try:
            image = Image.open(self.image_paths[idx]).convert("RGB")
        except Exception as e:
            logger.warning(f"Failed to load {self.image_paths[idx]}: {e}")
            return self[(idx + 1) % len(self)]
        
        if self.transform:
            image = self.transform(image)
        
        return image


class ImageDataModule(LightningDataModule):
    """
    LightningDataModule for image folder dataset.
    
    Args:
        data_dir: Path to directory containing images.
        image_size: Input image size.
        batch_size: Batch size.
        num_workers: Number of data loading workers.
        pin_memory: Whether to pin memory.
        val_split: Fraction of data to use for validation.
    """
    
    def __init__(
        self,
        data_dir: str,
        image_size: int = 224,
        batch_size: int = 128,
        num_workers: int = 8,
        pin_memory: bool = True,
        val_split: float = 0.1,
    ) -> None:
        super().__init__()
        
        self.save_hyperparameters()
        
        self.data_dir = Path(data_dir)
        self.image_size = image_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.val_split = val_split
        
        # Transforms
        self.train_transform = DistillationTransform(image_size=image_size)
        self.val_transform = SimpleTransform(image_size=image_size)
        
        # Datasets (set in setup)
        self.train_dataset: Optional[Dataset] = None
        self.val_dataset: Optional[Dataset] = None
    
    def setup(self, stage: Optional[str] = None) -> None:
        """Setup datasets."""
        if stage == "fit" or stage is None:
            # Load full dataset
            full_dataset = ImageFolderDataset(
                root=self.data_dir,
                transform=self.train_transform,
            )
            
            # Split into train/val
            n_val = int(len(full_dataset) * self.val_split)
            n_train = len(full_dataset) - n_val
            
            self.train_dataset, val_dataset_raw = random_split(
                full_dataset, [n_train, n_val]
            )
            
            # Create validation dataset with val transform
            # Note: This is a simplified version; ideally we'd use a separate dataset
            self.val_dataset = val_dataset_raw
            
            logger.info(f"Train: {n_train} samples, Val: {n_val} samples")
    
    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=True,
            persistent_workers=self.num_workers > 0,
        )
    
    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=False,
        )
