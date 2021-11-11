import torch
from torchvision import models
from byol_pytorch import BYOL
import pytorch_lightning as pl

LR         = 3e-4

class SelfSupervisedLearner(pl.LightningModule):
    def __init__(self, **kwargs):
        super().__init__()
        net = models.resnet50(pretrained=False)
        self.learner = BYOL(net, 
                            image_size = 224,
                            hidden_layer = 'avgpool',
                            projection_size = 256,
                            projection_hidden_size = 4096,
                            moving_average_decay = 0.99, )

    def forward(self, images):
        return self.learner(images)

    def get_representation(self, images): 
        representation, _ = self.learner(images, return_embedding=True)
        return representation 

    def training_step(self, images, _):
        loss = self.forward(images)
        self.log("train/loss", loss)
        return {'loss': loss}

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=LR)

    def on_before_zero_grad(self, _):
        if self.learner.use_momentum:
            self.learner.update_moving_average()
