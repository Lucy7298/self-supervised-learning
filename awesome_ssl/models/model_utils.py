import pytorch_lightning as ptl 
from awesome_ssl.models.trunk_models import models
from awesome_ssl.models.projection_heads import projection_heads
from torchvision import transforms

def build_backbone(model_configs): 
    backbone = models[model_configs['arch']]
    if model_configs.get('pytorch_pretrained', False): 
        if model_configs['resume_path'] == 'default': 
            model = backbone(pretrained=True)
        else: 
            raise Exception("Not implemented checkpoint loading yet")
    else: 
        model = backbone(pretrained=False)
    return model 

def build_projector(configs): 
    projector = projection_heads[configs['type']]
    return projection_heads(**configs.get('params', {}))

def build_transform(configs): 
    def build_one(one_config): 
        transf_name = configs['type']
        transf = transforms.__dict__[transf_name]
        return transf(configs.get('params', {}))
    
    all_transforms = list(map(build_one, configs))
    return transforms.compose(all_transforms)