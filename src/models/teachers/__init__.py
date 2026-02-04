"""Teacher model wrappers."""

from src.models.teachers.dinov2 import DINOv2Teacher
from src.models.teachers.base import BaseTeacher

__all__ = ["DINOv2Teacher", "BaseTeacher"]
