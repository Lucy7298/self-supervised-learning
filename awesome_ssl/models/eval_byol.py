from hydra.utils import instantiate
from omegaconf import DictConfig
import torch
import torch.nn.functional as F

import pytorch_lightning as pl
from torchmetrics.functional import accuracy
from pydoc import locate

class BYOL_Eval(pl.LightningModule): 
    def __init__(self, 
                 model_class : str,
                 weight_path : str, 
                 representation_size: int,
                 num_targets: int): 
        super().__init__()
        model = locate(model_class).load_from_checkpoint(weight_path)
        self.model = model
        self.model.eval()

        self.classifier = torch.nn.Linear(representation_size, num_targets)

    def training_step(self, batch, batch_idx): 
        X, y = batch
        prediction = self.classifier(self.model(X).detach())
        return F.cross_entropy(prediction, y)

    def validation_step(self, batch, batch_idx): 
        X, y = batch
        prediction = self.classifier(self.model(X))
        self.log("val/cross_entropy", F.cross_entropy(prediction, y))
        self.log("val/top-1", accuracy(prediction, y, top_k=1))
        self.log("val/top-5", accuracy(prediction, y, top_k=5))

    def configure_optimizers(self): 
        return torch.optim.SGD(self.classifier.parameters(), lr=0.1)