from typing import List, Optional
from pytorch_lightning.plugins import DDPPlugin
import hydra
from omegaconf import DictConfig

from awesome_ssl.models.model_utils import build_module
from pytorch_lightning.trainer.trainer import Trainer
from pytorch_lightning.loggers import WandbLogger  # newline 1
from hydra.utils import instantiate
from awesome_ssl.datasets.dataloader_utils import return_train_val_dataloaders
import os
import re 
import omegaconf
import wandb
import logging
from awesome_ssl.callbacks.metrics import InvarianceMetric
import pandas as pd 



def evaluate(config: DictConfig) -> Optional[float]:
    """Contains training pipeline.
    Instantiates all PyTorch Lightning objects from config.
    Args:
        config (DictConfig): Configuration composed by Hydra.
    Returns:
        Optional[float]: Metric score for hyperparameter optimization.
    """

    # initialize callback, trainer, datasets 
    callback = instantiate(config.callback)

    print(callback)
    trainer = Trainer(**config.trainer, 
                      callbacks=[callback])
    train_dataloader, val_dataloader = return_train_val_dataloaders(config, shuffle_train=False)

    # get training config, initialize model 
    model_config_path = os.path.join(config.train_dir, ".hydra/config.yaml")
    run_conf = omegaconf.OmegaConf.load(model_config_path)
    print(run_conf, run_conf.model.target)
    if "transforms" in run_conf: 
        model = build_module(run_conf.model, transforms=run_conf.transforms)
    else: 
        model = build_module(run_conf.model)

    all_data = []
    # get weights 
    weight_file = config.weight_path

    # load weight 
    print("loading from checkpoint", weight_file)
    model = model.load_from_checkpoint(os.path.join(config.train_dir, "checkpoints", weight_file))
    print("finished loading from checkpoint")
    # extract the epoch number
    data_folder = os.path.join(config.train_dir, config.output_dir)
    if not os.path.exists(data_folder): 
        os.mkdir(data_folder)
    epoch_num = int(re.match(r"epoch_([0-9]+).ckpt", weight_file).group(1))

    callback.initialize_examples(train_dataloader)
    trainer.validate(model, train_dataloader)
    additional_data = {
        "dataset": config.dataset_name, 
        "epoch": epoch_num, 
        "split": "train"
    }
    callback.save_data(os.path.join(data_folder, f"train_{config.dataset_name}_{epoch_num}.ckpt"), 
                       additional_data)

    callback.initialize_examples(val_dataloader)
    trainer.validate(model, val_dataloader)
    additional_data = {
        "dataset": config.dataset_name, 
        "epoch": epoch_num, 
        "split": "val", 
        "save_directory": os.getcwd()
    }
    callback.save_data(os.path.join(data_folder, f"val_{config.dataset_name}_{epoch_num}.ckpt"), 
                       additional_data)
        

    # return metric for sweeper 
    if config.get("optimized_metric"): 
        score = trainer.callback_metrics.get(config.get("optimized_metric"))
        logging.warn(f"score is {score}")
        return score 

