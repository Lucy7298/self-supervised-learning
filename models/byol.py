import .model_utils
import torch
import pytorch_lightning as pl
from pl_bolts.optimizers.lars_scheduling import LARSWrapper

class BYOL(pl.LightningModule): 
    def __init__(self, config): 
        super().__init__(self)
        self.config = config 
        self.online_encoder = torch.nn.Sequential([
            model_utils.build_backbone(config['backbone']['encoder']), 
            model_utils.build_projector(config['projection_head']['projection'])
        ])
        self.target_encoder = torch.nn.Sequential([
            model_utils.build_backbone(config['backbone']['encoder']), 
            model_utils.build_projector(config['projection_head']['projection'])
        ])

        for target_param in self.target_encoder.parameters(): 
            # do I need to copy the online encoder to the target encoder? 
            target_param.requires_grad = False 

        self.prediction_head = model_utils.build_projector(config['projection_head']['predictor'])
        self.loss = torch.nn.MSELoss()
        self.t = config.t

    def training_step(self, batch, batch_idx): 
        q_theta = self.online_encoder(batch)
        online_prediction = self.prediction_head(q_theta)

        # update the target network with the online network
        with torch.no_grad(): 
            for target_p, online_p in zip(self.target_encoder.parameters(), \
                                          self.online_encoder.parameters()): 
                target_p.data = self.t * target_p.data + (1 - self.t) * online_p.data

        # is detach call necessary? target network doesn't require grad anyways
        target = self.target_encoder(batch).detach() 
        prediction_norm = online_prediction / \
            torch.linalg.norm(online_prediction, dim=1, keepdim=True)
        target_norm = target / \
            torch.linalg.norm(target, dim=1, keepdim=True)
        return self.loss(prediction_norm, target_norm)

    def configure_optimizers(self): 
        # I don't know how to recreate their scheduler :( 
        optimizer = torch.optim.SGD([
                {'params': self.online_encoder.parameters()}, 
                {'params': self.prediction_head.parameters()}
            ], 
            lr=0.2, 
            weight_decay=1.5e-6)
        scheduler = LARSWrapper(optimizer, eta=1e-3)
        return {
            "lr_scheduler": scheduler, 
            "optimizer": optimizer
        }

        

