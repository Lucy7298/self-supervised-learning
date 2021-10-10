from typing import List, Optional

import hydra
from omegaconf import DictConfig
#from pytorch_lightning.loggers import LightningLoggerBase

from awesome_ssl.models.model_utils import build_module


def train(config: DictConfig) -> Optional[float]:
    """Contains training pipeline.
    Instantiates all PyTorch Lightning objects from config.
    Args:
        config (DictConfig): Configuration composed by Hydra.
    Returns:
        Optional[float]: Metric score for hyperparameter optimization.
    """

    model = build_module(config.model)
