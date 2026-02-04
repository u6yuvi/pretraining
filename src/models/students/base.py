"""Base student wrapper protocol."""

from abc import ABC, abstractmethod
from typing import Dict

import torch.nn as nn
from torch import Tensor


class StudentWrapper(nn.Module, ABC):
    """
    Abstract base class for student model wrappers.
    
    Students must expose spatial features (B, C, H, W) without pooling
    so they can be aligned with the teacher's spatial grid.
    """
    
    @abstractmethod
    def forward_features(self, x: Tensor) -> Dict[str, Tensor]:
        """
        Extract spatial features without pooling.
        
        Args:
            x: Input images (B, C, H, W).
            
        Returns:
            Dict with "features" tensor of shape (B, feature_dim, H_out, W_out).
        """
        pass
    
    @abstractmethod
    def feature_dim(self) -> int:
        """Return the feature dimension (number of channels)."""
        pass
    
    @abstractmethod
    def get_model(self) -> nn.Module:
        """Return the underlying model for export."""
        pass
    
    def forward(self, x: Tensor) -> Tensor:
        """Default forward returns spatial features."""
        return self.forward_features(x)["features"]
