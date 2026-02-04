"""General utility functions."""

import warnings
from functools import wraps
from typing import Any, Callable, Dict, Optional, Tuple

from omegaconf import DictConfig

from src.utils.pylogger import get_pylogger

log = get_pylogger(__name__)


def extras(cfg: DictConfig) -> None:
    """
    Apply optional utilities before training.
    
    Args:
        cfg: Hydra configuration.
    """
    # Disable warnings if requested
    if cfg.get("ignore_warnings"):
        log.info("Disabling Python warnings...")
        warnings.filterwarnings("ignore")

    # Print config if requested
    if cfg.get("print_config"):
        from omegaconf import OmegaConf
        log.info(f"Config:\n{OmegaConf.to_yaml(cfg)}")


def task_wrapper(task_func: Callable) -> Callable:
    """
    Wrapper for training task to handle exceptions gracefully.
    
    Args:
        task_func: The training function to wrap.
        
    Returns:
        Wrapped function with exception handling.
    """
    @wraps(task_func)
    def wrapper(cfg: DictConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        try:
            # Apply extras
            extras(cfg)
            
            # Run task
            metric_dict, object_dict = task_func(cfg)
            
        except Exception as e:
            log.exception(f"Training failed with exception: {e}")
            raise
            
        finally:
            # Cleanup
            log.info("Task finished!")
        
        return metric_dict, object_dict
    
    return wrapper


def get_metric_value(
    metric_dict: Dict[str, Any], 
    metric_name: Optional[str]
) -> Optional[float]:
    """
    Safely retrieve metric value for hyperparameter optimization.
    
    Args:
        metric_dict: Dictionary of metrics from training.
        metric_name: Name of the metric to retrieve.
        
    Returns:
        Metric value as float, or None if not found.
    """
    if not metric_name:
        log.info("Metric name not provided! Returning None.")
        return None

    if metric_name not in metric_dict:
        log.warning(
            f"Metric '{metric_name}' not found in metric_dict! "
            f"Available metrics: {list(metric_dict.keys())}"
        )
        return None

    metric_value = metric_dict[metric_name].item()
    log.info(f"Retrieved metric '{metric_name}': {metric_value}")

    return metric_value
