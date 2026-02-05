"""
Export trained student model for downstream fine-tuning.

Usage:
    python src/export.py checkpoint=path/to/checkpoint.ckpt
"""

from pathlib import Path
from typing import Dict

import hydra
import torch
from omegaconf import DictConfig

from src import utils

log = utils.get_pylogger(__name__)


def export_student_weights(
    checkpoint_path: str | Path,
    output_path: str | Path,
    student_prefix: str = "student.",
) -> None:
    """
    Export student model weights from a training checkpoint.
    
    Args:
        checkpoint_path: Path to Lightning checkpoint.
        output_path: Path to save exported weights.
        student_prefix: Prefix for student weights in state dict.
    """
    checkpoint_path = Path(checkpoint_path)
    output_path = Path(output_path)
    
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    log.info(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    
    state_dict = checkpoint.get("state_dict", {})
    
    # Extract student weights
    student_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith(student_prefix):
            new_key = key[len(student_prefix):]
            student_state_dict[new_key] = value
    
    if not student_state_dict:
        log.warning(f"No weights found with prefix '{student_prefix}'")
        # Try without prefix
        student_state_dict = {
            k: v for k, v in state_dict.items() 
            if not k.startswith("teacher.")
        }
    
    log.info(f"Exporting {len(student_state_dict)} weight tensors")
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(student_state_dict, output_path)
    
    log.info(f"Saved student weights to {output_path}")


def export_yolo_model(
    checkpoint_path: str | Path,
    output_path: str | Path,
    model_config: str = "yolov8s.yaml",
    backbone_prefix: str = "student._backbone.",
) -> None:
    """
    Export as Ultralytics YOLO model.
    
    Args:
        checkpoint_path: Path to Lightning checkpoint.
        output_path: Path to save YOLO model.
        model_config: YOLO model configuration file.
    """
    try:
        from ultralytics import YOLO
    except ImportError:
        raise ImportError("ultralytics required: pip install ultralytics")
    
    checkpoint_path = Path(checkpoint_path)
    output_path = Path(output_path)
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("state_dict", {})
    
    # Extract backbone weights from student
    student_state_dict: Dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        if key.startswith(backbone_prefix):
            # Remove "student._backbone." prefix
            new_key = key[len(backbone_prefix):]
            student_state_dict[new_key] = value

    if not student_state_dict:
        raise ValueError(
            f"No backbone weights found with prefix '{backbone_prefix}'. "
            "Expected keys like 'student._backbone.0.conv.weight'."
        )
    
    # Create YOLO model and load weights
    model = YOLO(model_config)

    # Map backbone keys to YOLO model keys
    # Example: _backbone.0.conv.weight -> model.0.conv.weight
    mapped_state_dict = {}
    for key, value in student_state_dict.items():
        mapped_state_dict[f"model.{key}"] = value

    missing, unexpected = model.model.load_state_dict(mapped_state_dict, strict=False)
    log.info(f"Loaded backbone weights into YOLO model.")
    if missing:
        log.info(f"Missing keys (expected for neck/head): {len(missing)}")
    if unexpected:
        log.warning(f"Unexpected keys while loading: {unexpected}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(output_path))

    log.info(f"Saved YOLO model to {output_path}")


@hydra.main(version_base="1.3", config_path="configs", config_name="export.yaml")
def main(cfg: DictConfig):
    """Hydra entry point for export."""
    export_format = cfg.get("export_format", "student_weights")

    if export_format == "yolo":
        export_yolo_model(
            checkpoint_path=cfg.checkpoint,
            output_path=cfg.output,
            model_config=cfg.get("yolo_model_config", "yolov8s.yaml"),
            backbone_prefix=cfg.get("yolo_backbone_prefix", "student._backbone."),
        )
    else:
        export_student_weights(
            checkpoint_path=cfg.checkpoint,
            output_path=cfg.output,
            student_prefix=cfg.get("student_prefix", "student."),
        )


if __name__ == "__main__":
    main()
