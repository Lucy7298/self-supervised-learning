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
from pytorch_lightning.callbacks import ModelCheckpoint
import os
import wandb
import logging



def train(config: DictConfig) -> Optional[float]:
    """Contains training pipeline.
    Instantiates all PyTorch Lightning objects from config.
    Args:
        config (DictConfig): Configuration composed by Hydra.
    Returns:
        Optional[float]: Metric score for hyperparameter optimization.
    """

    model = build_module(config.model)
    print(model)
    logger = []
    if "logger" in config:
        for _, lg_conf in config.logger.items():
            if "_target_" in lg_conf:
                log = hydra.utils.instantiate(lg_conf)
                print(log)
                logger.append(log)

    callbacks = []
    if "callbacks" in config:
        for _, cb_conf in config.callbacks.items():
            if "_target_" in cb_conf:
                print(f"Instantiating callback <{cb_conf._target_}>")
                callbacks.append(hydra.utils.instantiate(cb_conf))

    trainer = Trainer(**config.trainer, 
                      logger=logger, 
                      callbacks=callbacks)
    train_dataloader, val_dataloader = return_train_val_dataloaders(config)
    trainer.fit(model, train_dataloader, val_dataloader)
    wandb.finish()

    # return metric for sweeper 
    if config.get("optimized_metric"): 
        score = trainer.callback_metrics.get(config.get("optimized_metric"))
        logging.warn(f"score is {score}")
        return score 
