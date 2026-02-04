"""Base teacher protocol."""

from abc import ABC, abstractmethod
from typing import Tuple

import torch.nn as nn
from torch import Tensor


class BaseTeacher(nn.Module, ABC):
    """
    Abstract base class for teacher models.
    
    Teachers must:
        1. Be frozen (no gradients)
        2. Expose intermediate layer features
        3. Provide embed_dim and total_embed_dim properties
    """
    
    @property
    @abstractmethod
    def embed_dim(self) -> int:
        """Embedding dimension of a single block."""
        pass
    
    @property
    @abstractmethod
    def total_embed_dim(self) -> int:
        """Total embedding dimension (n_blocks * embed_dim)."""
        pass
    
    @abstractmethod
    def forward(self, x: Tensor) -> Tuple[Tensor, Tuple[int, int]]:
        """
        Extract spatial features from images.
        
        Args:
            x: Input images (B, C, H, W).
            
        Returns:
            Tuple of:
                - features: (B, H*W, total_embed_dim)
                - grid_size: (H_grid, W_grid)
        """
        pass
