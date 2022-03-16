from sklearn.pipeline import Pipeline
from torch.utils.data import DataLoader, Dataset
from pytorch_lightning.core.datamodule import LightningDataModule
import torch
from hydra.utils import instantiate
import os
from ffcv.loader import Loader, OrderOption
from ffcv.fields.basics import IntDecoder
from ffcv.transforms import ToTensor, Squeeze, Convert, ToTorchImage
from ffcv.fields.decoders import RandomResizedCropRGBImageDecoder
from ffcv.pipeline.pipeline_spec import PipelineSpec
from ffcv.fields import json
from awesome_ssl.models.model_utils import LINEAR_FIT_TRAIN_TRANFORM
import torchvision
import numpy as np

def make_loader(path, batch_size, num_workers, transforms_dict, additional_fields): 
    decoder = RandomResizedCropRGBImageDecoder((224, 224))
    pre_transforms = [ToTensor(), 
                      ToTorchImage(), 
                      Convert(torch.float32), 
                      torchvision.transforms.Normalize(0, 255)]
    pipelines = {}
    pipelines["image"] = PipelineSpec(source='image',
                                           decoder=decoder,  
                                           transforms=pre_transforms + LINEAR_FIT_TRAIN_TRANFORM.transforms)
    # # make a pipeline for each transform in transforms_dict 
    # # put global views first
    for idx, (k, v) in enumerate(transforms_dict["global"].items()): 
        pipelines[f"image_g{idx}"] = PipelineSpec(source='image',
                                                  decoder=decoder,
                                                  transforms=pre_transforms + v)

    for idx, (k, v) in enumerate(transforms_dict.get("local", {}).items()): 
        pipelines[f"image_l{idx}"] = PipelineSpec(source="image",
                                                  decoder=decoder,
                                                  transforms=pre_transforms + v)

    for fieldname, fieldtype in additional_fields.items(): 
        if fieldtype == 'label': 
            pipelines[fieldname] = [IntDecoder(), ToTensor(), Squeeze()]
    
    print("my pipelines", pipelines)

    distributed_flag = torch.distributed.is_initialized()
    print("is loader distributed:", distributed_flag )
    return Loader(path,
                batch_size=batch_size,
                num_workers=num_workers,
                order=OrderOption.RANDOM,
                pipelines=pipelines, 
                distributed=distributed_flag)

def make_batch_processor(transforms, additional_fields): 
    def process_batch(batch): 
        global_crops = batch[:len(transforms["global"])]
        local_crops = batch[len(global_crops): len(transforms.get("local", {}))]
        additional_data = []
        idx = len(global_crops) + len(local_crops)
        for fieldname, fieldtype  in additional_fields.items(): 
            data = batch[idx]
            if fieldtype == 'json': 
                data = json.unpack(data)
            additional_data.append(data)
        return global_crops, local_crops, *additional_data
    return process_batch

def return_train_val_dataloaders(config): 
    
    batch_size = config.batch_size
    if config.num_workers > 0: 
        num_workers = config.num_workers
    else: 
        num_workers = 24
    print(num_workers)
    
    _, ext = os.path.splitext(config.train_dataset.root)
    if ext == '.beton': 
        transforms = instantiate(config.transforms)
        train_dataloader = make_loader(config.train_dataset.root, 
                                       batch_size, 
                                       num_workers, 
                                       transforms, 
                                       config.train_dataset.additional_fields)
        val_dataloader = make_loader(config.val_dataset.root, 
                                     batch_size, 
                                     num_workers, 
                                     transforms, 
                                     config.val_dataset.additional_fields)
        train_batch_processor = make_batch_processor(transforms, 
                                                     config.train_dataset.additional_fields)
        val_batch_processor = make_batch_processor(transforms, 
                                                   config.val_dataset.additional_fields)
        
    else: 
        raise Exception("Unsupported dataloader - You can only use ffcv now")

    return train_dataloader, val_dataloader, train_batch_processor, val_batch_processor
