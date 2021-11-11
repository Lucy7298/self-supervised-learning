from awesome_ssl.models import model_utils
import torch
import torch.nn.functional as F

import pytorch_lightning as pl
from typing import Sequence, Optional, Union
from enum import Enum
import torchmetrics 
from torchmetrics.functional import accuracy

class LinearEvaluateConfig(Enum): 
    NoLinearEvaluate = 0 # supported
    LinearFitTrain = 1 # supported 
    LinearFitSeparate = 2 # not supported

class BYOL(pl.LightningModule): 
    def __init__(self,
                encoder_params: model_utils.ModuleConfig, 
                projector_params: model_utils.ModuleConfig, 
                predictor_params: model_utils.ModuleConfig,
                tau: float, 
                accumulate_n_batch: int, 
                optimizer_params: model_utils.ModuleConfig, 
                linear_evaluate: Union[int, LinearEvaluateConfig] = 0, 
                linear_evaluate_config: Optional[model_utils.ModuleConfig] = None): 
        super().__init__()
        self.save_hyperparameters()
        self.online_encoder = torch.nn.Sequential(
            model_utils.build_module(encoder_params), 
            model_utils.build_module(projector_params)
        )

        self.target_encoder = torch.nn.Sequential(
            model_utils.build_module(encoder_params), 
            model_utils.build_module(projector_params)
        )

        if isinstance(linear_evaluate, int): 
            linear_evaluate = LinearEvaluateConfig(linear_evaluate)
        self.linear_evaluate = linear_evaluate
        if self.linear_evaluate != LinearEvaluateConfig.NoLinearEvaluate: 
            self.classifier = model_utils.build_module(linear_evaluate_config)

        for target_param in self.target_encoder.parameters(): 
            target_param.requires_grad = False 

        self.prediction_head = model_utils.build_module(predictor_params)

        self.automatic_optimization = False
        self.optimizer_params = optimizer_params
        self.accumulate_n_batch = accumulate_n_batch
        self.t = tau
    
        self.train_accuracy = torchmetrics.Accuracy()

    def get_representation(self, x): 
        return self.prediction_head(self.online_encoder(x))

    def calculate_loss(self, online_prediction, target): 
        prediction_norm = torch.nn.functional.normalize(online_prediction)
        target_norm = torch.nn.functional.normalize(target)  
        return F.mse_loss(prediction_norm, target_norm)

    def training_step(self, batch, batch_idx): 
        enc_1, enc_2, target = batch
        with torch.no_grad(): 
            target_1 = self.target_encoder(enc_1).detach()
            target_2 = self.target_encoder(enc_2).detach()
        on_pred_1 = self.prediction_head(self.online_encoder(enc_1))
        on_pred_2 = self.prediction_head(self.online_encoder(enc_2))

        if self.linear_evaluate == LinearEvaluateConfig.NoLinearEvaluate: 
            opt_enc = self.optimizers()
        else: 
            opt_enc, opt_class = self.optimizers()

        loss_1 = self.calculate_loss(on_pred_1, target_1)
        loss_2 = self.calculate_loss(on_pred_2, target_2)
        loss = loss_1 + loss_2
        self.log("train/loss", loss)
        self.manual_backward(loss)
        if batch_idx % self.accumulate_n_batch == 0: 
            opt_enc.step()
            opt_enc.zero_grad()

        # complete update step for classifier without label leakage 
        if self.linear_evaluate == LinearEvaluateConfig.LinearFitTrain: 
            pred_1 = self.classifier(on_pred_1.detach())
            pred_2 = self.classifier(on_pred_2.detach())
            # should you do backpropogation over 
            # both predicted or one predicted? 
            class_loss = F.cross_entropy(pred_1, target) + F.cross_entropy(pred_2, target)
            self.log("train/class_loss", class_loss)
            # log metrics 
            with torch.no_grad(): 
                self.log("train/class_accuracy", self.train_accuracy(pred_1, target), on_step=False, on_epoch=True)

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
        if self.linear_evaluate == LinearEvaluateConfig.LinearFitTrain:
            # log metrics on linear evaluation on validation set 
            X, y = batch
            on_pred = self.prediction_head(self.online_encoder(X))
            pred = self.classifier(on_pred)
            loss = F.cross_entropy(pred, y)
            self.log("val/class_top1", accuracy(pred, y))
            self.log("val/class_top5", accuracy(pred, y, top_k=5))
            self.log("val/cross_entropy", loss)

    def configure_optimizers(self): 
        # encoder optimizer
        model_params = [
            {'params': self.online_encoder.parameters()}, 
            {'params': self.prediction_head.parameters()
        }]
        optimizer_enc = model_utils.build_optimizer(self.optimizer_params, model_params)
        if self.linear_evaluate == LinearEvaluateConfig.NoLinearEvaluate: 
            return {"optimizer": optimizer_enc}

        # hard-coding classifier optimizer for now
        else: 
            classifier_params = [{
                "params": self.classifier.parameters()
            }]
            optimizer_class = torch.optim.SGD(classifier_params, lr=0.01)

            return (
                {"optimizer": optimizer_enc}, 
                {"optimizer": optimizer_class}
            )

        

