from torch.utils.data import DataLoader, Dataset
from pytorch_lightning.core.datamodule import LightningDataModule
from torch.cuda import device_count
from hydra.utils import instantiate

def return_train_val_dataloaders(config): 
    
    batch_size = config.batch_size
    if config.num_workers > 0: 
        num_workers = config.num_workers
    else: 
        num_workers = 4*device_count()
    train_dataset = instantiate(config.train_dataset)
    val_dataset = instantiate(config.val_dataset)
    train_dataloader = DataLoader(train_dataset, 
                          batch_size=batch_size, 
                          shuffle=True, 
                          drop_last=True, 
                          num_workers=num_workers)

    val_dataloader =  DataLoader(val_dataset, 
                          batch_size=batch_size, 
                          shuffle=False, 
                          drop_last=False, 
                          num_workers = num_workers)

    return train_dataloader, val_dataloader
