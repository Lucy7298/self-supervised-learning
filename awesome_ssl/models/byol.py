from awesome_ssl.models import model_utils
import torch
import pytorch_lightning as pl
from typing import Sequence

class BYOL(pl.LightningModule): 
    def __init__(self,
                encoder_params: model_utils.ModuleConfig, 
                projector_params: model_utils.ModuleConfig, 
                predictor_params: model_utils.ModuleConfig,
                tau: float, 
                accumulate_n_batch: int, 
                transform_1: Sequence[model_utils.ModuleConfig], 
                transform_2: Sequence[model_utils.ModuleConfig], 
                loss_module: model_utils.ModuleConfig, 
                optimizer_params: model_utils.ModuleConfig): 
        super().__init__()
        self.online_encoder = torch.nn.Sequential(
            model_utils.build_module(encoder_params), 
            model_utils.build_module(projector_params)
        )

        self.target_encoder = torch.nn.Sequential(
            model_utils.build_module(encoder_params), 
            model_utils.build_module(projector_params)
        )

        for target_param in self.target_encoder.parameters(): 
            target_param.requires_grad = False 

        self.prediction_head = model_utils.build_module(predictor_params)
        self.loss = model_utils.build_module(loss_module)

        self.automatic_optimization = False

        self.transform_1 = model_utils.build_augmentations(transform_1)
        self.transform_2 = model_utils.build_augmentations(transform_2)
        self.optimizer_params = optimizer_params
        self.accumulate_n_batch = accumulate_n_batch
        self.t = tau

    def calculate_loss(self, online_prediction, target): 
        prediction_norm = online_prediction / \
            torch.linalg.norm(online_prediction, dim=1, keepdim=True)
        target_norm = target / \
            torch.linalg.norm(target, dim=1, keepdim=True)   
        return self.loss(prediction_norm, target_norm)   

    def training_step(self, batch, batch_idx): 
        sample, target, fname = batch
        print("shape of sample", sample.shape)
        enc_1 = self.transform_1(sample)
        enc_2 = self.transform_2(sample)
        print("shape of sample after encoding", enc_1.shape)
        on_pred_1= self.prediction_head(self.online_encoder(enc_1))
        target_1 = self.target_encoder(enc_1).detach() 
        on_pred_2 = self.prediction_head(self.online_encoder(enc_2))
        target_2 = self.target_encoder(enc_2).detach()

        #complete update step 
        opt = self.optimizers()
        loss_1 = self.calculate_loss(on_pred_1, target_1)
        loss_2 = self.calculate_loss(on_pred_2, target_2)
        loss = loss_1 + loss_2
        loss.backward()

        if batch_idx % self.accumulate_n_batch == 0: 
            opt.step()
            opt.zero_grad()

        # update the target network with the online network
        with torch.no_grad(): 
            for target_p, online_p in zip(self.target_encoder.parameters(), \
                                          self.online_encoder.parameters()): 
                target_p.data = self.t * target_p.data + (1 - self.t) * online_p.data

    def configure_optimizers(self): 
        # I don't know how to recreate their scheduler :( 
        model_params = [{'params': self.online_encoder.parameters()}, 
                  {'params': self.prediction_head.parameters()}]
        optimizer = model_utils.build_optimizer(self.optimizer_params, model_params)
        return {
            "optimizer": optimizer
        }

        

