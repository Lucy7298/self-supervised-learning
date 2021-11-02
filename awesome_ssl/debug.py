from typing import List, Optional
from pytorch_lightning.plugins import DDPPlugin
import hydra
from omegaconf import DictConfig
#from pytorch_lightning.loggers import LightningLoggerBase

from awesome_ssl.models.model_utils import build_module
from pytorch_lightning.trainer.trainer import Trainer
from pytorch_lightning.loggers import WandbLogger  # newline 1
from hydra.utils import instantiate
from awesome_ssl.datasets.dataloader_utils import return_train_val_dataloaders
import os


def debug(config: DictConfig) -> Optional[float]:
    """Contains training pipeline.
    Instantiates all PyTorch Lightning objects from config.
    Args:
        config (DictConfig): Configuration composed by Hydra.
    Returns:
        Optional[float]: Metric score for hyperparameter optimization.
    """

    model = build_module(config.model)
    print(model)
    trainer = Trainer(**config.trainer)
    train_dataloader, val_dataloader = return_train_val_dataloaders(config)
    trainer.fit(model, train_dataloader, val_dataloader)
