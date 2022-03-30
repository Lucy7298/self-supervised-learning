from torch.utils.data import DataLoader, ConcatDataset, Subset, Dataset
from pytorch_lightning.core.datamodule import LightningDataModule
from torch.cuda import device_count
import torch
from hydra.utils import instantiate
from hydra import compose
import os

from zmq import device
from awesome_ssl.datasets.custom_datasets import * 
CONFIG_PATH = "/mnt/nfs/home/yunxingl/self-supervised-learning/configs"

def return_train_val_dataloaders(config, shuffle_train=True): 
    
    batch_size = config.batch_size
    
    num_workers = config.num_workers 
    if num_workers > 0: 
        num_workers = config.num_workers
    else: 
        num_workers = 4 * device_count()
        print(f"using {num_workers} workers")
    
    train_dataset = make_dataset(config.train_dataset)
    val_dataset = make_dataset(config.val_dataset)
    train_dataloader = DataLoader(train_dataset, 
                        batch_size=batch_size, 
                        shuffle=shuffle_train, 
                        drop_last=True, 
                        num_workers=num_workers)

    val_dataloader =  DataLoader(val_dataset, 
                        batch_size=batch_size, 
                        shuffle=False, 
                        drop_last=False, 
                        num_workers = num_workers)

    return train_dataloader, val_dataloader


def concat_datasets(relative_path_configs): 
    all_dsets = []
    for relative_path in relative_path_configs: 
        if isinstance(relative_path, str): 
            dataset = list(instantiate(compose(relative_path)).values())[0]
        elif isinstance(relative_path, Dataset):
            dataset = relative_path
        else:  
            dataset = relative_path.module # if it is dictconfig, should have been instantiated already 
        assert isinstance(dataset, Dataset)
        print(len(dataset))
        all_dsets.append(dataset)

    return ConcatDataset(all_dsets)

def subset_dataset(relative_path, num_examples, random_seed=0):
    import random
    random.seed(random_seed)
    if isinstance(relative_path, str): 
        dataset = list(instantiate(compose(relative_path)).values())[0]
    else: 
        dataset = relative_path
    indices = random.sample(list(range(len(dataset))), num_examples)
    return Subset(dataset, indices)


def make_dataset(ds_config): 
    print(ds_config)
    if hasattr(ds_config, "func_object"): 
        command = ds_config.func_object.replace("\\", "")
        print(command)
        return eval(command)
    else: 
        return instantiate(ds_config)