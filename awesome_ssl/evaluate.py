from typing import List, Optional
from pytorch_lightning.plugins import DDPPlugin
import hydra
from omegaconf import DictConfig

from awesome_ssl.models.model_utils import build_module
from pytorch_lightning.trainer.trainer import Trainer
from pytorch_lightning.loggers import WandbLogger  # newline 1
from hydra.utils import instantiate
from hydra import compose 
from torch.utils.data import DataLoader 
import os
import re 
import omegaconf
import wandb
import logging
from awesome_ssl.callbacks.metrics import InvarianceMetric
from awesome_ssl.callbacks.relative_distance import RelativeDistance
import pandas as pd 
from pl_bolts.models.self_supervised.ssl_finetuner import SSLFineTuner
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint


callbacks = [
    [InvarianceMetric(), RelativeDistance()], 
    [ "invar_data", "confusion"]
]
num_callbacks = 2

dset_config_paths = [
    ("train_dataset/imagenet.yaml", "val_dataset/imagenet_val.yaml", "imagenet", 1000), 
    ("train_dataset/imagenette.yaml", "val_dataset/imagenette_val.yaml", "imagenette", 10), 
    ("train_dataset/imagewoof.yaml", "val_dataset/imagewoof_val.yaml", "imagewoof", 10)
]

def make_dataloader(config_relpath): 
    config = instantiate(compose(config_name=config_relpath))
    dset = instantiate(config)
    if "train_dataset" in dset: 
        dset = dset["train_dataset"]
    elif "val_dataset" in dset: 
        dset = dset["val_dataset"]
    print(dset)
    return DataLoader(dset, 
                      batch_size=50, 
                      shuffle=True, 
                      drop_last=False, 
                      num_workers=4)

def load_model(config): 
    # get training config, initialize model 
    model_config_path = os.path.join(config.train_dir, ".hydra/config.yaml")
    run_conf = omegaconf.OmegaConf.load(model_config_path)
    print(run_conf, run_conf.model.target)
    if "transforms" in run_conf: 
        model = build_module(run_conf.model, transforms=run_conf.transforms)
    else: 
        model = build_module(run_conf.model)

    # get weights 
    weight_file = config.weight_path

    # load weight 
    print("loading from checkpoint", weight_file)
    model = model.load_from_checkpoint(os.path.join(config.train_dir, "checkpoints", weight_file))
    print("finished loading from checkpoint")
    return model 


def run_callbacks(config): 
    for train_path, val_path, dset_name, _ in dset_config_paths: 
        train_dataloader = make_dataloader(train_path)
        val_dataloader = make_dataloader(val_path)
        
        # initialize callback, trainer, datasets 
        trainer = Trainer(**config.trainer, 
                        callbacks=callbacks[0], 
                        enable_checkpointing=False)
        my_callbacks = callbacks[0][:num_callbacks]
        # get training config, initialize model 
        model = load_model(config)

        # extract the epoch number
        for output_dir in callbacks[1]: 
            data_folder = os.path.join(config.train_dir, output_dir)
            if not os.path.exists(data_folder): 
                os.mkdir(data_folder)
        
        epoch_num = int(re.match(r"epoch_([0-9]+).ckpt", config.weight_path).group(1))

        for callback in my_callbacks: 
            callback.initialize_examples(train_dataloader)
        trainer.validate(model, train_dataloader)
        additional_data = {
            "dataset": dset_name, 
            "epoch": epoch_num, 
            "split": "train",
            "save_directory": os.getcwd()
        }

        for callback, output_dir in zip(my_callbacks, callbacks[1]): 
            data_folder = os.path.join(config.train_dir, output_dir)
            callback.save_data(os.path.join(data_folder, f"train_{dset_name}_{epoch_num}.ckpt"), 
                            additional_data)

        for callback in my_callbacks:
            callback.initialize_examples(val_dataloader)

        trainer.validate(model, val_dataloader)
        additional_data = {
            "dataset": dset_name, 
            "epoch": epoch_num, 
            "split": "val", 
            "save_directory": os.getcwd()
        }

        for callback, output_dir in zip(my_callbacks, callbacks[1]): 
            data_folder = os.path.join(config.train_dir, output_dir)
            callback.save_data(os.path.join(data_folder, f"val_{dset_name}_{epoch_num}.ckpt"), 
                            additional_data)

def run_linear_evaluation(config): 
    model = load_model(config)
    for train_path, val_path, dset_name, num_classes in dset_config_paths:  
        save_dir = os.path.join(config.train_dir, "linear_fit")
        ckpt_dir = os.path.join(config.save_dir, "linear_fit_checkpoints")
        logger = TensorBoardLogger(save_dir=save_dir, version=1, name=dset_name)
        callback = ModelCheckpoint(dirpath=ckpt_dir, 
                                   save_top_k = 1, 
                                   monitor="test_acc", 
                                   mode='max', 
                                   filename="{epoch}-{test_acc:.2f}")
        trainer = pl.Trainer(gpus=-1, logger=logger, callbacks=callback)
        train_loader = make_dataloader(train_path)
        val_loader = make_dataloader(val_path)
        module = SSLFineTuner(model, 
                    in_features = 2048, 
                    num_classes = num_classes)
        trainer.fit(module, train_loader, val_loader)
        trainer.validate(module, [train_loader, val_loader])
        

            

def evaluate(config: DictConfig) -> Optional[float]:
    """Contains training pipeline.
    Instantiates all PyTorch Lightning objects from config.
    Args:
        config (DictConfig): Configuration composed by Hydra.
    Returns:
        Optional[float]: Metric score for hyperparameter optimization.
    """

    run_callbacks(config)
    run_linear_evaluation(config)

