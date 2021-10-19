from typing import List, Optional
from pytorch_lightning.plugins import DDPPlugin
import hydra
from omegaconf import DictConfig
#from pytorch_lightning.loggers import LightningLoggerBase

from awesome_ssl.models.model_utils import build_module
from pytorch_lightning.trainer.trainer import Trainer
from pytorch_lightning.loggers import WandbLogger  # newline 1
import os


def train(config: DictConfig) -> Optional[float]:
    """Contains training pipeline.
    Instantiates all PyTorch Lightning objects from config.
    Args:
        config (DictConfig): Configuration composed by Hydra.
    Returns:
        Optional[float]: Metric score for hyperparameter optimization.
    """

    model = build_module(config.model)
    datamodule = build_module(config.dataset)

    wandb_logger = WandbLogger(project="BYOL") 
    wandb_logger.log_hyperparams({'output_directory': os.getcwd()})
    trainer = Trainer(**config.trainer, 
                      logger=wandb_logger, 
                      plugins=DDPPlugin(find_unused_parameters=False))
    trainer.fit(model, datamodule=datamodule)
