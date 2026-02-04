"""Training methods (LightningModules)."""

from src.methods.distillation_module import DistillationModule
from src.methods.losses import DistillationLoss, MSELoss, CosineLoss

__all__ = [
    "DistillationModule",
    "DistillationLoss",
    "MSELoss",
    "CosineLoss",
]
