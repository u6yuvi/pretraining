"""
YOLO model wrapper for distillation.

Wraps Ultralytics YOLO models and extracts backbone features.
"""

from __future__ import annotations

import logging
from typing import Any, Dict

import torch.nn as nn
from torch import Tensor

from src.models.students.base import StudentWrapper

logger = logging.getLogger(__name__)


class YOLOWrapper(StudentWrapper):
    """
    Wrapper for Ultralytics YOLO models.
    
    Extracts the backbone features (before SPPF/neck) for distillation.
    Supports YOLOv5, YOLOv6, YOLOv8, YOLO11, and YOLO12.
    
    Args:
        model_name: YOLO model name (e.g., "yolov8s.yaml").
        pretrained: Whether to load pretrained weights.
    """
    
    def __init__(
        self,
        model_name: str = "yolov8s.yaml",
        pretrained: bool = False,
    ) -> None:
        super().__init__()
        
        # Import ultralytics
        try:
            from ultralytics import YOLO
        except ImportError:
            raise ImportError(
                "ultralytics is required for YOLOWrapper. "
                "Install with: pip install ultralytics"
            )
        
        # Load model
        logger.info(f"Loading YOLO model: {model_name}")
        self._yolo = YOLO(model_name)
        
        # Enable gradients
        self._enable_gradients()
        
        # Set to training mode
        self._yolo.model.train()
        
        # Extract backbone
        self._backbone, self._feature_dim = self._get_backbone()
        
        logger.info(f"YOLOWrapper initialized: feature_dim={self._feature_dim}")
    
    def _enable_gradients(self) -> None:
        """Enable gradients on all model parameters."""
        for param in self._yolo.model.parameters():
            param.requires_grad = True
    
    def _get_backbone(self) -> tuple[nn.Sequential, int]:
        """Extract backbone from YOLO model."""
        from ultralytics.nn.modules.block import C2f, C3, SPPF
        from ultralytics.nn.modules.head import Classify
        
        seq = self._yolo.model.model
        assert isinstance(seq, nn.Sequential)
        
        for idx, module in enumerate(seq):
            if idx == 0:
                continue
            
            prev_module = seq[idx - 1]
            
            # YOLOv5: C3 before SPPF
            if type(prev_module) is C3 and type(module) is SPPF:
                return seq[:idx], prev_module.cv3.conv.out_channels
            
            # YOLOv6: Sequential before SPPF
            if type(prev_module) is nn.Sequential and type(module) is SPPF:
                return seq[:idx], prev_module[-1].conv.out_channels
            
            # YOLOv8/11/12: C2f before SPPF or Classify
            if type(prev_module) is C2f and type(module) in (SPPF, Classify):
                return seq[:idx], prev_module.cv2.conv.out_channels
        
        raise ValueError("Could not determine YOLO backbone structure")
    
    def forward_features(self, x: Tensor) -> Dict[str, Tensor]:
        """Extract backbone features."""
        features = self._backbone(x)
        return {"features": features}
    
    def feature_dim(self) -> int:
        return self._feature_dim
    
    def get_model(self) -> Any:
        """Return the full YOLO model for export."""
        return self._yolo
