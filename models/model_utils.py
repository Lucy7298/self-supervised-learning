import pytorch_lightning as ptl 
from .trunk_models import models
from .projection_heads import projection_heads

def build_backbone(model_configs): 
    backbone = models[model_configs['arch']]
    if model_configs['pytorch_pretrained']: 
        if model_configs['resume_path'] == 'default': 
            model = backbone(pretrained=True)
        else: 
            raise Exception("Not implemented checkpoint loading yet")
    else: 
        model = backbone(pretrained=False)
    return model 

def build_projector(configs): 
    projector = projection_heads[configs['type']]
    return projection_heads(**configs['params'])
