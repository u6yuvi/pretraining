"""
DINOv2 Teacher wrapper.

Loads a DINOv2 Vision Transformer, freezes it, and exposes intermediate layer
features for distillation.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Tuple

import torch
import torch.nn as nn
from torch import Tensor

from src.models.teachers.base import BaseTeacher

logger = logging.getLogger(__name__)


# Supported DINOv2 models from torch.hub
DINOV2_MODELS = {
    "dinov2_vits14": {"embed_dim": 384, "patch_size": 14},
    "dinov2_vitb14": {"embed_dim": 768, "patch_size": 14},
    "dinov2_vitl14": {"embed_dim": 1024, "patch_size": 14},
    "dinov2_vitg14": {"embed_dim": 1536, "patch_size": 14},
    # With registers
    "dinov2_vits14_reg": {"embed_dim": 384, "patch_size": 14},
    "dinov2_vitb14_reg": {"embed_dim": 768, "patch_size": 14},
    "dinov2_vitl14_reg": {"embed_dim": 1024, "patch_size": 14},
    "dinov2_vitg14_reg": {"embed_dim": 1536, "patch_size": 14},
}


class DINOv2Teacher(BaseTeacher):
    """
    DINOv2 Vision Transformer teacher for knowledge distillation.
    
    Loads a pretrained DINOv2 model, freezes all parameters, and provides
    access to intermediate layer features for spatial distillation.
    
    Args:
        model_name: Name of the DINOv2 model (e.g., "dinov2_vitb14").
        n_blocks: Number of intermediate blocks to extract features from.
        weights_path: Optional path to custom weights file.
    """
    
    def __init__(
        self,
        model_name: str = "dinov2_vitb14",
        n_blocks: int = 2,
        weights_path: str | Path | None = None,
    ) -> None:
        super().__init__()
        
        if model_name not in DINOV2_MODELS:
            raise ValueError(
                f"Unknown model: {model_name}. "
                f"Supported: {list(DINOV2_MODELS.keys())}"
            )
        
        self.model_name = model_name
        self.n_blocks = n_blocks
        self._model_info = DINOV2_MODELS[model_name]
        
        # Load model from torch.hub
        logger.info(f"Loading DINOv2 teacher: {model_name}")
        self.model = torch.hub.load(
            "facebookresearch/dinov2",
            model_name,
            pretrained=weights_path is None,
        )
        
        # Load custom weights if provided
        if weights_path is not None:
            weights_path = Path(weights_path)
            if not weights_path.exists():
                raise FileNotFoundError(f"Weights not found: {weights_path}")
            state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)
            self.model.load_state_dict(state_dict)
            logger.info(f"Loaded custom weights from {weights_path}")
        
        # Make teacher deterministic (remove stochastic depth)
        self._make_deterministic()
        
        # Freeze all parameters
        self._freeze()
        
        # Set to eval mode
        self.model.eval()
        
        logger.info(
            f"Teacher ready: embed_dim={self.embed_dim}, "
            f"n_blocks={n_blocks}, total_dim={self.total_embed_dim}"
        )
    
    @property
    def embed_dim(self) -> int:
        """Embedding dimension of a single block."""
        return self._model_info["embed_dim"]
    
    @property
    def patch_size(self) -> int:
        """Patch size of the ViT."""
        return self._model_info["patch_size"]
    
    @property
    def total_embed_dim(self) -> int:
        """Total embedding dimension (n_blocks * embed_dim)."""
        return self.n_blocks * self.embed_dim
    
    def _make_deterministic(self) -> None:
        """Remove stochastic depth (DropPath) for deterministic outputs."""
        for block in self.model.blocks:
            if hasattr(block, "drop_path1"):
                block.drop_path1 = nn.Identity()
            if hasattr(block, "drop_path2"):
                block.drop_path2 = nn.Identity()
            if hasattr(block, "sample_drop_ratio"):
                block.sample_drop_ratio = 0.0
        logger.debug("Made teacher deterministic")
    
    def _freeze(self) -> None:
        """Freeze all parameters."""
        for param in self.model.parameters():
            param.requires_grad = False
        logger.debug("Froze all teacher parameters")
    
    @torch.no_grad()
    def forward(self, x: Tensor) -> Tuple[Tensor, Tuple[int, int]]:
        """
        Extract intermediate layer features from the teacher.
        
        Args:
            x: Input images of shape (B, C, H, W).
            
        Returns:
            Tuple of:
                - features: (B, H*W, n_blocks * embed_dim)
                - grid_size: (H_grid, W_grid)
        """
        # Get intermediate layer features
        features_list = self.model.get_intermediate_layers(
            x, n=self.n_blocks, reshape=True
        )
        
        # Get spatial dimensions from last layer
        _, _, h_grid, w_grid = features_list[-1].shape
        
        # Resize all feature maps to same spatial size and concatenate
        aligned_features = []
        for feat in features_list:
            _, _, h, w = feat.shape
            if h != h_grid or w != w_grid:
                feat = nn.functional.interpolate(
                    feat,
                    size=(h_grid, w_grid),
                    mode="bilinear",
                    align_corners=False,
                )
            aligned_features.append(feat)
        
        # Concatenate: (B, n_blocks * D, H, W)
        x = torch.cat(aligned_features, dim=1)
        
        # Flatten spatial dims: (B, D, H, W) -> (B, H*W, D)
        x = x.permute(0, 2, 3, 1).flatten(start_dim=1, end_dim=2)
        
        return x, (h_grid, w_grid)
