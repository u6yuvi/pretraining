"""
Distillation LightningModule.

Main training module that orchestrates teacher, student, projection head,
and loss computation.
"""

from __future__ import annotations

import logging
import math
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from lightning import LightningModule
from torch import Tensor
from torch.optim import AdamW, Optimizer
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

from src.methods.losses import DistillationLoss
from src.models.components.projection_head import ProjectionHead
from src.models.students.base import StudentWrapper
from src.models.teachers.base import BaseTeacher

logger = logging.getLogger(__name__)


class DistillationModule(LightningModule):
    """
    PyTorch Lightning module for DINOv2 → Student distillation.
    
    Transfers knowledge from a frozen teacher to a student model
    by matching spatial features using MSE loss.
    
    Args:
        teacher: Teacher model (frozen, provides target features).
        student: Student model wrapper.
        optimizer: Optimizer (partial from Hydra).
        scheduler: LR scheduler (partial from Hydra).
        n_projection_layers: Number of projection head layers.
        projection_hidden_dim: Hidden dim of projection head.
        use_mixup: Whether to apply mixup augmentation.
        lr_scale_method: "linear" or "sqrt" for batch size scaling.
        reference_batch_size: Reference batch size for LR scaling.
    """
    
    def __init__(
        self,
        teacher: BaseTeacher,
        student: StudentWrapper,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler.LRScheduler] = None,
        n_projection_layers: int = 1,
        projection_hidden_dim: int = 2048,
        use_mixup: bool = True,
        lr_scale_method: str = "sqrt",
        reference_batch_size: int = 256,
    ) -> None:
        super().__init__()
        
        # Save hyperparameters (excluding models)
        self.save_hyperparameters(ignore=["teacher", "student"])
        
        # --- Teacher (frozen) ---
        self.teacher = teacher
        
        # --- Student (trainable) ---
        self.student = student
        
        # --- Projection head ---
        self.projection_head = ProjectionHead(
            in_dim=student.feature_dim(),
            out_dim=teacher.total_embed_dim,
            n_layers=n_projection_layers,
            hidden_dim=projection_hidden_dim,
        )
        
        # --- Loss ---
        self.criterion = DistillationLoss()
        
        # --- Config ---
        self.use_mixup = use_mixup
        self.lr_scale_method = lr_scale_method
        self.reference_batch_size = reference_batch_size
        
        logger.info(
            f"DistillationModule:\n"
            f"  Teacher dim: {teacher.total_embed_dim}\n"
            f"  Student dim: {student.feature_dim()}\n"
            f"  Projection: {student.feature_dim()} -> {teacher.total_embed_dim}"
        )
    
    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        """Forward through student."""
        return self.student.forward_features(x)
    
    @torch.no_grad()
    def _forward_teacher(self, x: Tensor) -> tuple[Tensor, tuple[int, int]]:
        """Forward through frozen teacher."""
        return self.teacher(x)
    
    def _forward_student(
        self,
        x: Tensor,
        teacher_h: int,
        teacher_w: int,
    ) -> Tensor:
        """Forward through student with projection and alignment."""
        # Get student features: (B, C, H, W)
        features = self.student.forward_features(x)["features"]
        
        # Project to teacher dim: (B, D_teacher, H, W)
        features = self.projection_head(features)
        
        # Resize to teacher grid
        features = F.interpolate(
            features,
            size=(teacher_h, teacher_w),
            mode="bilinear",
            align_corners=False,
        )
        
        # Flatten: (B, D, H, W) -> (B, H*W, D)
        features = features.permute(0, 2, 3, 1)
        features = features.flatten(start_dim=1, end_dim=2)
        
        return features
    
    def _mixup(self, x: Tensor) -> Tensor:
        """Apply mixup augmentation."""
        lam = torch.empty(1, device=x.device).uniform_(0.0, 1.0).item()
        index = torch.randperm(x.size(0), device=x.device)
        return lam * x + (1.0 - lam) * x[index]
    
    def training_step(self, batch: Any, batch_idx: int) -> Tensor:
        """Training step."""
        # Handle different batch formats
        if isinstance(batch, dict):
            images = batch.get("image", batch.get("images"))
        elif isinstance(batch, (tuple, list)):
            images = batch[0]
        else:
            images = batch
        
        # Optional mixup
        if self.use_mixup:
            images = self._mixup(images)
        
        # Teacher forward
        teacher_features, (teacher_h, teacher_w) = self._forward_teacher(images)
        
        # Student forward
        student_features = self._forward_student(images, teacher_h, teacher_w)
        
        # Loss
        loss = self.criterion(teacher_features, student_features)
        
        # Log
        self.log("train/loss", loss, prog_bar=True, sync_dist=True)
        
        return loss
    
    def validation_step(self, batch: Any, batch_idx: int) -> Tensor:
        """Validation step."""
        if isinstance(batch, dict):
            images = batch.get("image", batch.get("images"))
        elif isinstance(batch, (tuple, list)):
            images = batch[0]
        else:
            images = batch
        
        # Teacher forward
        teacher_features, (teacher_h, teacher_w) = self._forward_teacher(images)
        
        # Student forward (no mixup in val)
        student_features = self._forward_student(images, teacher_h, teacher_w)
        
        # Loss
        loss = self.criterion(teacher_features, student_features)
        
        self.log("val/loss", loss, prog_bar=True, sync_dist=True)
        
        return loss
    
    def configure_optimizers(self) -> Dict[str, Any]:
        """Configure optimizer and scheduler."""
        # Only optimize student and projection head
        params = (
            list(self.student.parameters()) + 
            list(self.projection_head.parameters())
        )
        
        # Instantiate optimizer
        optimizer = self.hparams.optimizer(params=params)
        
        if self.hparams.scheduler is None:
            return {"optimizer": optimizer}
        
        # Instantiate scheduler
        scheduler = self.hparams.scheduler(optimizer=optimizer)
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val/loss",
                "interval": "epoch",
                "frequency": 1,
            },
        }
    
    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        """Remove teacher from checkpoint (it's frozen/pretrained)."""
        state_dict = checkpoint.get("state_dict", {})
        checkpoint["state_dict"] = {
            k: v for k, v in state_dict.items()
            if not k.startswith("teacher.")
        }
    
    def on_load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        """Restore teacher weights when loading."""
        teacher_state = {
            f"teacher.{k}": v 
            for k, v in self.teacher.state_dict().items()
        }
        checkpoint["state_dict"].update(teacher_state)
