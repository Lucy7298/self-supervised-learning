from typing import List, Optional

import hydra
from omegaconf import DictConfig
#from pytorch_lightning.loggers import LightningLoggerBase

from awesome_ssl.models.model_utils import build_module
from pytorch_lightning.trainer.trainer import Trainer
from pytorch_lightning.loggers import WandbLogger  # newline 1
import os


def debug(config: DictConfig) -> Optional[float]:
    """Contains debugging pipeline.
    Instantiates all PyTorch Lightning objects from config.
    Args:
        config (DictConfig): Configuration composed by Hydra.
    Returns:
        Optional[float]: Metric score for hyperparameter optimization.
    """

    model = build_module(config.model)
    datamodule = build_module(config.dataset)
    trainer = Trainer(**config.trainer)
    trainer.fit(model, datamodule=datamodule)
