"""Student model wrappers."""

from src.models.students.base import StudentWrapper
from src.models.students.yolo import YOLOWrapper
from src.models.students.generic import GenericWrapper

__all__ = ["StudentWrapper", "YOLOWrapper", "GenericWrapper"]
