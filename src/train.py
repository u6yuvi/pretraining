"""
Main training script with Hydra configuration.

Usage:
    python src/train.py                              # Default config
    python src/train.py student=yolov8n             # Override student
    python src/train.py experiment=yolov8s_dinov2   # Full experiment
    python src/train.py trainer=ddp trainer.devices=4  # Multi-GPU
"""

from typing import List, Tuple

import hydra
import lightning as L
import torch
from lightning import Callback, LightningDataModule, LightningModule, Trainer
from lightning.pytorch.loggers import Logger
from omegaconf import DictConfig

from src import utils

log = utils.get_pylogger(__name__)


@utils.task_wrapper
def train(cfg: DictConfig) -> Tuple[dict, dict]:
    """Main training function.
    
    Instantiates all components from config and runs training.
    
    Args:
        cfg: Hydra configuration.
        
    Returns:
        Tuple of (metric_dict, object_dict).
    """
    # Set seed for reproducibility
    if cfg.get("seed"):
        L.seed_everything(cfg.seed, workers=True)

    # Instantiate datamodule
    log.info(f"Instantiating datamodule <{cfg.data._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)

    # Instantiate model (distillation module)
    log.info(f"Instantiating model <{cfg.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(cfg.model)

    # Instantiate loggers
    log.info("Instantiating loggers...")
    logger: List[Logger] = utils.instantiate_loggers(cfg.get("logger"))

    # Instantiate callbacks
    log.info("Instantiating callbacks...")
    callbacks: List[Callback] = utils.instantiate_callbacks(cfg.get("callbacks"))

    # Instantiate trainer
    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(
        cfg.trainer, 
        callbacks=callbacks, 
        logger=logger
    )

    # Store objects for logging
    object_dict = {
        "cfg": cfg,
        "datamodule": datamodule,
        "model": model,
        "trainer": trainer,
        "callbacks": callbacks,
        "logger": logger,
    }

    # Log hyperparameters
    if logger:
        log.info("Logging hyperparameters!")
        utils.log_hyperparameters(object_dict)

    # Compile model if requested (PyTorch 2.0+)
    if cfg.get("compile"):
        log.info("Compiling model with torch.compile()...")
        model = torch.compile(model)

    # Train
    if cfg.get("train"):
        log.info("Starting training!")
        trainer.fit(
            model=model, 
            datamodule=datamodule,
            ckpt_path=cfg.get("ckpt_path")
        )

    train_metrics = trainer.callback_metrics

    # Test
    if cfg.get("test"):
        log.info("Starting testing!")
        ckpt_path = trainer.checkpoint_callback.best_model_path
        if ckpt_path == "":
            log.warning("Best ckpt not found! Using current weights for testing...")
            ckpt_path = None
        trainer.test(model=model, datamodule=datamodule, ckpt_path=ckpt_path)
        log.info(f"Best ckpt path: {ckpt_path}")

    test_metrics = trainer.callback_metrics

    # Merge metrics
    metric_dict = {**train_metrics, **test_metrics}

    return metric_dict, object_dict


@hydra.main(version_base="1.3", config_path="configs", config_name="train.yaml")
def main(cfg: DictConfig):
    """Hydra entry point."""
    # Train the model
    metric_dict, _ = train(cfg)

    # Return metric for hyperparameter optimization
    metric_value = utils.get_metric_value(
        metric_dict=metric_dict, 
        metric_name=cfg.get("optimized_metric")
    )

    return metric_value


if __name__ == "__main__":
    main()
