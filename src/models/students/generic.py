"""Generic student wrapper for custom backbones."""

from __future__ import annotations

from typing import Any, Dict

import torch.nn as nn
from torch import Tensor

from src.models.students.base import StudentWrapper


class GenericWrapper(StudentWrapper):
    """
    Generic wrapper for CNN-like backbones.
    
    Works with models that have a backbone/features attribute,
    or that directly output feature maps.
    
    Args:
        model: The backbone model to wrap.
        feature_dim: Output feature dimension (channels).
        feature_layer: Optional name of the layer to extract features from.
    """
    
    def __init__(
        self,
        model: nn.Module,
        feature_dim: int,
        feature_layer: str | None = None,
    ) -> None:
        super().__init__()
        self._model = model
        self._feature_dim = feature_dim
        
        # Find feature extractor
        if feature_layer is not None:
            if not hasattr(model, feature_layer):
                raise ValueError(f"Model has no attribute '{feature_layer}'")
            self._extractor = getattr(model, feature_layer)
        elif hasattr(model, "features"):
            self._extractor = model.features
        elif hasattr(model, "backbone"):
            self._extractor = model.backbone
        else:
            self._extractor = model
    
    def forward_features(self, x: Tensor) -> Dict[str, Tensor]:
        features = self._extractor(x)
        return {"features": features}
    
    def feature_dim(self) -> int:
        return self._feature_dim
    
    def get_model(self) -> nn.Module:
        return self._model
