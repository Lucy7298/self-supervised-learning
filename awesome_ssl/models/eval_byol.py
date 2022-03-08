from hydra.utils import instantiate
from omegaconf import DictConfig
import torch
import torch.nn.functional as F

import pytorch_lightning as pl
from torchmetrics.functional import accuracy
from pydoc import locate
import re, os
from typing import Optional
import omegaconf
from awesome_ssl.models.model_utils import build_module

class BYOL_Eval(pl.LightningModule): 
    def __init__(self, 
                 train_dir: str, 
                 weight_file: str,
                 representation_size: int,
                 num_targets: int, 
                 learning_rate: float, 
                 transforms): # transforms parameters does not matter
        super().__init__()
        self.save_hyperparameters()
        model_config_path = os.path.join(train_dir, ".hydra/config.yaml")
        run_conf = omegaconf.OmegaConf.load(model_config_path)
        print(run_conf, run_conf.model.target)
        if "transforms" in run_conf: 
            model = build_module(run_conf.model, transforms=run_conf.transforms)
        else: 
            model = build_module(run_conf.model)

        # load weight 
        print("loading from checkpoint", weight_file)
        model = model.load_from_checkpoint(os.path.join(train_dir, "checkpoints", weight_file))
        self.model = model.eval()

        self.classifier = torch.nn.Linear(representation_size, num_targets)
        self.lr = learning_rate

        # log which epoch this model is for
        m = re.match(r"epoch(=|_)(?P<epoch>\d+)(-step=(?P<step>\d+))?.ckpt", weight_file)
        self.save_hyperparameters({"epoch_trained": int(m.group('epoch'))})

    def training_step(self, batch, batch_idx): 
        X, y = batch
        prediction = self.classifier(self.model.get_representation(X).detach())
        loss = F.cross_entropy(prediction, y)
        self.log("train/loss", loss)
        return loss

    def validation_step(self, batch, batch_idx): 
        X, y = batch
        prediction = self.classifier(self.model.get_representation(X))
        loss = F.cross_entropy(prediction, y)
        self.log("val_loss", loss)
        self.log("val/top-1", accuracy(prediction, y, top_k=1))
        self.log("val/top-5", accuracy(prediction, y, top_k=5))
        return loss 

    def configure_optimizers(self): 
        return torch.optim.SGD(self.classifier.parameters(), lr=self.lr)