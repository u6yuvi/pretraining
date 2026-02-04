"""Model wrappers for teachers and students."""

from src.models.teachers import DINOv2Teacher
from src.models.students import StudentWrapper, YOLOWrapper
from src.models.components import ProjectionHead

__all__ = [
    "DINOv2Teacher",
    "StudentWrapper",
    "YOLOWrapper",
    "ProjectionHead",
]
