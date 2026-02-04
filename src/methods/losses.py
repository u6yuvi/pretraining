"""Loss functions for distillation."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


class DistillationLoss(nn.Module):
    """
    MSE loss for knowledge distillation.
    
    Computes mean squared error between teacher and student features.
    """
    
    def __init__(self) -> None:
        super().__init__()
        self.mse = nn.MSELoss()
    
    def forward(
        self,
        teacher_features: Tensor,
        student_features: Tensor,
    ) -> Tensor:
        """
        Args:
            teacher_features: (B, N, D)
            student_features: (B, N, D)
            
        Returns:
            Scalar MSE loss.
        """
        return self.mse(teacher_features, student_features)


# Alias for clarity
MSELoss = DistillationLoss


class CosineLoss(nn.Module):
    """
    Cosine similarity loss for distillation.
    
    Focuses on direction rather than magnitude.
    """
    
    def __init__(self) -> None:
        super().__init__()
        self.cosine = nn.CosineSimilarity(dim=-1)
    
    def forward(
        self,
        teacher_features: Tensor,
        student_features: Tensor,
    ) -> Tensor:
        """
        Args:
            teacher_features: (B, N, D)
            student_features: (B, N, D)
            
        Returns:
            Mean of (1 - cosine_similarity).
        """
        sim = self.cosine(teacher_features, student_features)
        return (1.0 - sim).mean()


class SmoothL1Loss(nn.Module):
    """Smooth L1 loss, less sensitive to outliers."""
    
    def __init__(self, beta: float = 1.0) -> None:
        super().__init__()
        self.loss = nn.SmoothL1Loss(beta=beta)
    
    def forward(
        self,
        teacher_features: Tensor,
        student_features: Tensor,
    ) -> Tensor:
        return self.loss(teacher_features, student_features)
