from awesome_ssl.models import model_utils
import torch
torch.autograd.set_detect_anomaly(True)
import torch.nn.functional as F

import pytorch_lightning as pl
from typing import Sequence, Optional, Union
from enum import Enum
import torchmetrics 
from torchmetrics.functional import accuracy
from omegaconf import DictConfig
from hydra.utils import instantiate
from torchvision import transforms as T

class TrainStage(Enum): 
    PRETRAIN = 0 
    LINEAR_FIT = 1

DEFAULT_TRAIN_AUG = torch.nn.Sequential(
    T.RandomApply(
        [T.ColorJitter(0.8, 0.8, 0.8, 0.2)],
        p = 0.3
    ),
    T.RandomGrayscale(p=0.2),
    T.RandomHorizontalFlip(),
    T.RandomApply(
        [T.GaussianBlur((3, 3), (1.0, 2.0))],
        p = 0.2
    ),
    T.RandomResizedCrop((224, 224)),
    T.Normalize(
        mean=torch.tensor([0.485, 0.456, 0.406]),
        std=torch.tensor([0.229, 0.224, 0.225])),
)

LINEAR_FIT_TRAIN_TRANFORM = T.Compose([
        T.RandomHorizontalFlip(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225])
    ])
    
LINEAR_FIT_VAL_TRANFORM = T.Normalize(mean=[0.485, 0.456, 0.406],
                                      std=[0.229, 0.224, 0.225])

class SimCLR(pl.LightningModule): 
    def __init__(self,
                encoder_params: model_utils.ModuleConfig, 
                projector_params: model_utils.ModuleConfig, 
                accumulate_n_batch: int, 
                optimizer_params: model_utils.ModuleConfig, 
                temperature: float, 
                transform_1 : Optional[DictConfig] = None, 
                transform_2 : Optional[DictConfig] = None,
                eval_interval: Optional[int] = -1, #linear evaluate after linear evaluate epochs
                #if < 0, don't linear evaluate 
                linear_evaluate_config: Optional[model_utils.ModuleConfig] = None, 
                randominit_target: Optional[bool] = True): 
        super().__init__()
        self.save_hyperparameters()
        self.encoder = model_utils.build_module(encoder_params)
        self.prediction_head = model_utils.build_module(projector_params)
        self.eval_interval = eval_interval
        if self.eval_interval > 0: 
            self.classifier = model_utils.build_module(linear_evaluate_config)

        self.optimizer_params = optimizer_params

        if transform_1 is not None: 
            print("instantiating transform 1")
            self.transform_1 = instantiate(transform_1)
            print(self.transform_1)
        else: 
            print("loading default for transform 1")
            self.transform_1 = DEFAULT_TRAIN_AUG
        
        if transform_2 is not None: 
            print("instantiating transform 2")
            self.transform_2 = instantiate(transform_2)
            print(self.transform_2)
        else: 
            print("loading default for transform 2")
            self.transform_2 = DEFAULT_TRAIN_AUG
    
        self.train_stage = TrainStage.PRETRAIN
        self.temperature = temperature 

    def get_representation(self, x): 
        return self.encoder(x)

    def on_train_epoch_start(self): 
        if self.eval_interval > 0: 
            if self.current_epoch % self.eval_interval == 0: 
                print(f"epoch {self.current_epoch}: doing linear fit")
                self.encoder = self.encoder.eval()
                self.prediction_head = self.prediction_head.eval() 
                self.train_stage = TrainStage.LINEAR_FIT
            else: 
                print(f"epoch {self.current_epoch}: doing pretrain")
                self.encoder = self.encoder.train()
                self.prediction_head = self.prediction_head.train()
                self.train_stage = TrainStage.PRETRAIN

    def get_pretrain_loss(self, enc_1, enc_2, stage): 
        enc_1 = self.prediction_head(self.encoder(enc_1))
        enc_2 = self.prediction_head(self.encoder(enc_2))
        N, _ = enc_1.shape
        enc_1 = F.normalize(enc_1, dim=1, eps=1.0e-4)
        enc_2 =  F.normalize(enc_2, dim=1, eps=1.0e-4)
        # Yo what even is this loss
        all_encodings = torch.cat((enc_1, enc_2), axis=0)
        # pairwise similarity 
        similarities = all_encodings @ all_encodings.T / self.temperature
        mask = torch.eye(2*N, dtype=torch.bool, device=self.device)
        similarities = similarities[~mask].view(2*N, -1)
        labels = torch.cat(((torch.arange(N, device=self.device) + N - 1), \
                             torch.arange(N, device=self.device)), dim=0)
        loss = F.cross_entropy(similarities, labels, reduction='mean')
        self.log(f"{stage}/pretrain_loss", loss)
        return loss
    
    def get_linear_fit_loss(self, image, label, stage): 
        with torch.no_grad(): 
            image = self.get_representation(image).detach()
        pred = self.classifier(image)
        loss = F.cross_entropy(pred, label)
        with torch.no_grad(): 
            self.log(f"{stage}/linear_top1", accuracy(pred, label))
            self.log(f"{stage}/linear_top5", accuracy(pred, label, top_k=5))
            self.log(f"{stage}/linear_cross_entropy", loss)
        return loss

    def training_step(self, batch, batch_idx): 
        X, y = batch
        if self.eval_interval < 0: 
            opt_enc = self.optimizers()
        else: 
            opt_enc, opt_class = self.optimizers()
            
        if self.train_stage == TrainStage.PRETRAIN: 
            with torch.no_grad(): 
                enc_1, enc_2 = self.transform_1(X), self.transform_2(X)
            # pass through network 
            return self.get_pretrain_loss(enc_1, enc_2, "train")

        # complete update step for classifier without label leakage 
        elif self.train_stage == TrainStage.LINEAR_FIT: 
            with torch.no_grad(): 
                X = LINEAR_FIT_TRAIN_TRANFORM(X)
            return self.get_linear_fit_loss(X, y, "train")


    def validation_step(self, batch, batch_idx):
        X, y = batch
        if self.train_stage == TrainStage.LINEAR_FIT:
            with torch.no_grad(): 
                X = LINEAR_FIT_VAL_TRANFORM(X)
            # log metrics on linear evaluation on validation set 
            self.get_linear_fit_loss(X, y, "val")

        elif self.train_stage == TrainStage.PRETRAIN: 
            enc_1, enc_2 = self.transform_1(X), self.transform_2(X)
            self.get_pretrain_loss(enc_1, enc_2, "val")

    def configure_optimizers(self): 
        # encoder optimizer
        model_params = [
            {'params': self.encoder.parameters()}, 
            {'params': self.prediction_head.parameters()}
        ]
        optimizer_enc = model_utils.build_optimizer(self.optimizer_params, model_params)
        if self.eval_interval < 0: 
            return {"optimizer": optimizer_enc}

        else: 
            classifier_params = [{
                "params": self.classifier.parameters()
            }]
            optimizer_class = torch.optim.SGD(classifier_params, lr=0.02)

            return (
                {"optimizer": optimizer_enc}, 
                {"optimizer": optimizer_class}
            )

        

