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
                optimizer_params: model_utils.ModuleConfig, 
                transforms: DictConfig = None, #set transforms, transform_1, or transform_2
                transform_1: DictConfig = None, 
                transform_2: DictConfig = None, 
                eval_interval: Optional[int] = -1, #linear evaluate after linear evaluate epochs
                #if < 0, don't linear evaluate 
                linear_evaluate_config: Optional[model_utils.ModuleConfig] = None, 
                randominit_target: Optional[bool] = True): 
        super().__init__()
        self.save_hyperparameters()
        self.online_encoder = torch.nn.Sequential(
            model_utils.build_module(encoder_params), 
            model_utils.build_module(projector_params)
        )
        if randominit_target: 
            print("randomly initializing target encoder...")
            self.target_encoder = torch.nn.Sequential(
                model_utils.build_module(encoder_params), 
                model_utils.build_module(projector_params)
            )
        else: 
            print("copying target encoder...")
            self.target_encoder = copy.deepcopy(self.online_encoder)
        self.eval_interval = eval_interval
        if self.eval_interval > 0: 
            self.classifier = model_utils.build_module(linear_evaluate_config)

        for target_param in self.target_encoder.parameters(): 
            target_param.requires_grad = False 

        self.prediction_head = model_utils.build_module(predictor_params)

        self.automatic_optimization = False
        self.optimizer_params = optimizer_params
        self.accumulate_n_batch = accumulate_n_batch
        self.t = tau

        self.transform_1, self.transform_2 = build_transform(transform_1, transform_2, transforms)

    
        self.train_stage = TrainStage.PRETRAIN

    def get_representation(self, x, return_projection=False):
        if return_projection:  
            return self.online_encoder(x)
        else: 
            return self.online_encoder[0](x)

    def get_projection(self, x): 
        return self.prediction_head(self.online_encoder(x))

    def calculate_loss(self, online_prediction, target): 
        prediction_norm = torch.nn.functional.normalize(online_prediction, dim=-1)
        target_norm = torch.nn.functional.normalize(target, dim=-1)  
        return torch.sum((prediction_norm - target_norm)**2, dim=-1)

    def on_train_epoch_start(self): 
        if self.eval_interval > 0: 
            if self.current_epoch % self.eval_interval == 0: 
                print(f"epoch {self.current_epoch}: doing linear fit and pretrain")
                # self.online_encoder = self.online_encoder.eval()
                # self.prediction_head = self.prediction_head.eval() 
                self.train_stage = TrainStage.LINEAR_FIT
            else: 
                print(f"epoch {self.current_epoch}: doing pretrain")
                # self.online_encoder = self.online_encoder.train()
                # self.prediction_head = self.prediction_head.train()
                self.train_stage = TrainStage.PRETRAIN

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
    
    def get_linear_fit_loss(self, image, label, stage): 
        with torch.no_grad(): 
            on_pred = self.get_representation(image).detach()
        pred = self.classifier(on_pred)
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

        

