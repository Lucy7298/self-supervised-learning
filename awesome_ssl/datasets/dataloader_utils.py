from torch.utils.data import DataLoader, Dataset
from pytorch_lightning.core.datamodule import LightningDataModule
from torch.cuda import device_count
import torch
from hydra.utils import instantiate
import os
# from ffcv.loader import Loader, OrderOption
# from ffcv.fields.basics import IntDecoder
# from ffcv.transforms import ToTensor, Squeeze, Convert

# def make_loader(path, batch_size, num_workers, image_pipeline): 
#     pipelines = {
#         "image": image_pipeline + [Convert(torch.float32)], # should probably change according to config
#         "label": [IntDecoder(), ToTensor(), Squeeze()]
#     }
#     distributed_flag = device_count() > 0
#     print("device count: ", device_count(), "distributed: ", distributed_flag)
#     return Loader(path,
#                 batch_size=batch_size,
#                 num_workers=num_workers,
#                 order=OrderOption.RANDOM,
#                 pipelines=pipelines, 
#                 distributed=distributed_flag)

def return_train_val_dataloaders(config, shuffle_train=True): 
    
    batch_size = config.batch_size
    if config.num_workers > 0: 
        num_workers = config.num_workers
    else: 
        num_workers = max(4*device_count(), 1)
    print(num_workers)
    
    _, ext = os.path.splitext(config.train_dataset.root)
    # if ext == '.beton': 
    #     train_transforms = instantiate(config.train_dataset.pipelines)
    #     test_transforms = instantiate(config.val_dataset.pipelines)
    #     train_dataloader = make_loader(config.train_dataset.root, 
    #                                    batch_size, 
    #                                    num_workers, 
    #                                    train_transforms.image)
    #     val_dataloader = make_loader(config.val_dataset.root, 
    #                                  batch_size, 
    #                                  num_workers, 
    #                                  test_transforms.image)
    # else: 
    train_dataset = instantiate(config.train_dataset)
    val_dataset = instantiate(config.val_dataset)
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
