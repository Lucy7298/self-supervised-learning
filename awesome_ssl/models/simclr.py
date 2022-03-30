from awesome_ssl.models import model_utils
from awesome_ssl.models.model_utils import LINEAR_FIT_TRAIN_TRANFORM, LINEAR_FIT_VAL_TRANFORM, build_transform
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
import math

class TrainStage(Enum): 
    PRETRAIN = 0 
    LINEAR_FIT = 1

class SimCLR(pl.LightningModule): 
    def __init__(self,
                encoder_params: model_utils.ModuleConfig, 
                projector_params: model_utils.ModuleConfig, 
                accumulate_n_batch: int, 
                optimizer_params: model_utils.ModuleConfig, 
                temperature: float, 
                transforms: DictConfig = None, #set transforms, transform_1, or transform_2
                transform_1: DictConfig = None, 
                transform_2: DictConfig = None, 
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

        self.automatic_optimization = False
        self.accumulate_n_batch = accumulate_n_batch
        self.optimizer_params = optimizer_params
        self.transform_1, self.transform_2 = build_transform(transform_1, transform_2, transforms)
        print(torch.distributed.is_initialized())
        self.train_stage = TrainStage.PRETRAIN
        self.temperature = temperature 

    def get_representation(self, x): 
        return self.encoder(x)

    def get_projection(self, x): 
        return self.prediction_head(self.encoder(x))

    def on_train_epoch_start(self): 
        if self.eval_interval > 0: 
            if self.current_epoch % self.eval_interval == 0: 
                print(f"epoch {self.current_epoch}: doing linear fit")
                # self.encoder = self.encoder.eval()
                # self.prediction_head = self.prediction_head.eval() 
                self.train_stage = TrainStage.LINEAR_FIT
            else: 
                print(f"epoch {self.current_epoch}: doing pretrain")
                # self.encoder = self.encoder.train()
                # self.prediction_head = self.prediction_head.train()
                self.train_stage = TrainStage.PRETRAIN

    def get_pretrain_loss(self, enc_1, enc_2, stage): 
        # select indexes 
        enc_1 = self.get_projection(enc_1)
        enc_1 = F.normalize(enc_1, dim=1)
        if torch.distributed.is_initialized(): 
            enc_1_all = model_utils.concat_all_gather(self, enc_1)
        else: 
            enc_1_all = enc_1

        enc_2 = self.get_projection(enc_2)
        enc_2 =  F.normalize(enc_2, dim=1)
        if torch.distributed.is_initialized(): 
            enc_2_all = model_utils.concat_all_gather(self, enc_2)
        else: 
            enc_2_all = enc_2

        batch_samples = torch.cat((enc_1, enc_2), axis=0)
        all_samples = torch.cat((enc_1_all, enc_2_all), axis=0)

        # calculate positive similarity 
        pos = torch.einsum("ij,ij->i", enc_1, enc_2).unsqueeze(-1)
        pos = torch.exp(torch.cat((pos, pos), axis=0) / self.temperature)

        # calculate denominator 
        neg = torch.exp((batch_samples @ all_samples.t()) / self.temperature)
        neg = torch.sum(neg, dim=-1)
        row_sub = torch.Tensor(neg.shape).fill_(math.e ** (1 / self.temperature)).to(neg.device)
        neg = torch.clamp(neg - row_sub, min=1.0e-6)
        loss = -1*torch.log(pos / (neg + 1.0e-6))
        self.log(f"{stage}/pretrain_loss", loss.mean())
        return loss.mean()
    
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
            
        with torch.no_grad(): 
            enc_1, enc_2 = self.transform_1(X), self.transform_2(X)
        # pass through network 
        loss = self.get_pretrain_loss(enc_1, enc_2, "train")
        self.manual_backward(loss)
        if batch_idx % self.accumulate_n_batch == 0: 
            opt_enc.step()
            opt_enc.zero_grad()

        # complete update step for classifier without label leakage 
        if self.train_stage == TrainStage.LINEAR_FIT: 
            with torch.no_grad(): 
                X = LINEAR_FIT_TRAIN_TRANFORM(X)
            class_loss = self.get_linear_fit_loss(X, y, "train")
            self.manual_backward(class_loss)
            # will update linear classifier every step 
            opt_class.step()
            opt_class.zero_grad()


    def validation_step(self, batch, batch_idx):
        X, y = batch
        if self.train_stage == TrainStage.LINEAR_FIT:
            with torch.no_grad(): 
                X = LINEAR_FIT_VAL_TRANFORM(X)
            # log metrics on linear evaluation on validation set 
            self.get_linear_fit_loss(X, y, "val")

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