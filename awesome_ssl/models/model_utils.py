from pydoc import locate
from dataclasses import dataclass, field
from typing import Any, Sequence, Dict, MutableMapping, MutableSequence
import torch
import torchvision
from omegaconf import OmegaConf
from torchvision import transforms as T

@dataclass
class ModuleConfig: 
    target: str 
    args: Sequence[Any] = field(default_factory=list)
    kwargs: Dict[str, Any] = field(default_factory=dict)

def autocast_inputs(config): 
    if isinstance(config, ModuleConfig): 
        return config
    elif isinstance(config, MutableMapping): 
        return ModuleConfig(**config)
    else: 
        raise Exception(f"could not autocast input config {config}")

def build_module(configs: ModuleConfig, *args, **kwargs): 
    configs = autocast_inputs(configs)
    module = locate(configs.target)
    return module(*configs.args, *args, **configs.kwargs, **kwargs) 

def build_augmentations(configs: Sequence[ModuleConfig]):
    # need to convert lists in config to tuples 
    all_transforms = []
    for conf in configs: 
        conf = OmegaConf.to_container(conf)
        all_transforms.append(build_module(conf))
    if all([isinstance(i, torch.nn.Module) for i in all_transforms]): 
        return torch.nn.Sequential(*all_transforms)
    else: 
        return torchvision.transforms.Compose(all_transforms)

def build_optimizer(configs: ModuleConfig, model_params): 
    module = locate(configs.target)
    args = configs.get('args', [])
    kwargs = configs.get('kwargs', {})
    return module(model_params, *args, **kwargs)

@torch.no_grad()
def concat_all_gather(pl_module, x): 
    x = pl_module.all_gather(x)
    ws, B, D = x.shape
    x = x.view(-1, D)
    return x

LINEAR_FIT_TRAIN_TRANFORM = T.Compose([
        T.RandomHorizontalFlip(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225])
    ])

LINEAR_FIT_VAL_TRANFORM = T.Normalize(mean=[0.485, 0.456, 0.406],
                                      std=[0.229, 0.224, 0.225])

