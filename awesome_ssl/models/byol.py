from awesome_ssl.models import model_utils
from awesome_ssl.models.model_utils import LINEAR_FIT_TRAIN_TRANFORM, LINEAR_FIT_VAL_TRANFORM, build_transform
import torch
import torch.nn.functional as F

import pytorch_lightning as pl
from typing import Sequence, Optional, Union
from enum import Enum
import torchmetrics 
from torchmetrics.functional import accuracy
from omegaconf import DictConfig
from hydra.utils import instantiate
from torchvision import transforms as T
import copy

class TrainStage(Enum): 
    PRETRAIN = 0 
    LINEAR_FIT = 1

class BYOL(pl.LightningModule): 
    def __init__(self,
                encoder_params: model_utils.ModuleConfig, 
                projector_params: model_utils.ModuleConfig, 
                predictor_params: model_utils.ModuleConfig,
                tau: float, 
                accumulate_n_batch: int, 
                optimizer_params: model_utils.ModuleConfig): 
        super().__init__()
        self.save_hyperparameters()
        self.online_encoder = torch.nn.Sequential(
            model_utils.build_module(encoder_params), 
            model_utils.build_module(projector_params)
        )

        self.target_encoder = copy.deepcopy(self.online_encoder)

        for target_param in self.target_encoder.parameters(): 
            target_param.requires_grad = False 

        self.prediction_head = model_utils.build_module(predictor_params)

        self.optimizer_params = optimizer_params
        self.accumulate_n_batch = accumulate_n_batch
        self.t = tau
    
    def get_representation(self, x):
        return self.online_encoder[0](x)


    def get_projection(self, x): 
        return self.prediction_head(self.online_encoder(x))

    def calculate_loss(self, online_prediction, target): 
        prediction_norm = torch.nn.functional.normalize(online_prediction, dim=-1)
        target_norm = torch.nn.functional.normalize(target, dim=-1)  
        return torch.sum((prediction_norm - target_norm)**2, dim=-1)

    def get_pretrain_loss(self, enc_1, enc_2, stage): 
        with torch.no_grad(): 
            target_1 = self.target_encoder(enc_1).detach()
            target_2 = self.target_encoder(enc_2).detach()
        on_pred_1 = self.get_projection(enc_1)
        on_pred_2 = self.get_projection(enc_2)
        loss_1 = self.calculate_loss(on_pred_1, target_2)
        loss_2 = self.calculate_loss(on_pred_2, target_1)
        loss = (loss_1 + loss_2).mean()
        self.log(f"{stage}/pretrain_loss", loss)
        return loss

    def training_step(self, batch, batch_idx): 
        img_1, img_2, y = batch
        if self.eval_interval < 0: 
            opt_enc = self.optimizers()
        else: 
            opt_enc, opt_class = self.optimizers()
            
        if self.train_stage == TrainStage.PRETRAIN: 
            with torch.no_grad(): 
                enc_1, enc_2 = self.transform_1(X), self.transform_2(X)
            loss = self.get_pretrain_loss(enc_1, enc_2, "train")
            self.manual_backward(loss)
            if batch_idx % self.accumulate_n_batch == 0: 
                opt_enc.step()
                opt_enc.zero_grad()

        # complete update step for classifier without label leakage 
        elif self.train_stage == TrainStage.LINEAR_FIT: 
            with torch.no_grad(): 
                X = LINEAR_FIT_TRAIN_TRANFORM(X)
            class_loss = self.get_linear_fit_loss(X, y, "train")
            self.manual_backward(class_loss)
            # will update linear classifier every step 
            opt_class.step()
            opt_class.zero_grad()

        # update the target network with the online network
        with torch.no_grad(): 
            for target_p, online_p in zip(self.target_encoder.parameters(), \
                                          self.online_encoder.parameters()): 
                target_p.data = self.t * target_p.data + (1 - self.t) * online_p.data

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
            {'params': self.online_encoder.parameters()}, 
            {'params': self.prediction_head.parameters()
        }]
        optimizer_enc = model_utils.build_optimizer(self.optimizer_params, model_params)
        if self.eval_interval < 0: 
            return {"optimizer": optimizer_enc}

        # hard-coding classifier optimizer for now
        else: 
            classifier_params = [{
                "params": self.classifier.parameters()
            }]
            optimizer_class = torch.optim.SGD(classifier_params, lr=0.02)

            return (
                {"optimizer": optimizer_enc}, 
                {"optimizer": optimizer_class}
            )

        

