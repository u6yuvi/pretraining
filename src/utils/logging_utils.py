"""Logging utilities for experiment tracking."""

from typing import Any, Dict

from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig, OmegaConf

from src.utils.pylogger import get_pylogger

log = get_pylogger(__name__)


def log_hyperparameters(object_dict: Dict[str, Any]) -> None:
    """
    Log hyperparameters to all loggers.
    
    Args:
        object_dict: Dictionary containing cfg, model, datamodule, etc.
    """
    hparams = {}

    cfg: DictConfig = object_dict.get("cfg", {})
    model = object_dict.get("model")
    trainer = object_dict.get("trainer")

    if not trainer:
        log.warning("Trainer not found in object_dict! Skipping hyperparameter logging.")
        return

    # Convert config to dict
    if cfg:
        hparams["cfg"] = OmegaConf.to_container(cfg, resolve=True)

    # Add model info
    if model:
        hparams["model"] = {
            "class": model.__class__.__name__,
            "num_params": sum(p.numel() for p in model.parameters()),
            "num_trainable_params": sum(
                p.numel() for p in model.parameters() if p.requires_grad
            ),
        }

    # Log to all loggers
    for logger in trainer.loggers:
        if isinstance(logger, Logger):
            logger.log_hyperparams(hparams)
