from random import Random
from pytorch_lightning.callbacks import Callback
import torchvision.transforms as T 
from awesome_ssl.augmentations.crop_and_shift import RandomCenterCrop
import torch
import torch.nn.functional as F
from torchvision.datasets import ImageFolder
from torchmetrics import MeanMetric
from collections import defaultdict
import pandas as pd 
import pickle 

class SaveEmbeddings(Callback): 
    def __init__(self): 
        super().__init__()
        self.reset_state()

    def reset_state(self): 
        self.representations = []
        self.projections = []
        self.ys = []
        
    def initialize_examples(self, eval_dataloader):
        pass 

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        X, y = batch
        representation = pl_module.get_representation(X)
        projection = pl_module.get_projection(X)
        self.representations.append(representation.cpu())
        self.projections.append(projection.cpu())
        self.ys.append(y)

    def save_data(self, output_path, additional_data): 
        data = {"representations": torch.cat(self.representations), 
                "projections": torch.cat(self.projections), 
                "ys": torch.cat(self.ys)}
        data.update(additional_data)
        with open(output_path, 'wb') as handle: 
            pickle.dump(data, handle)
        self.reset_state()