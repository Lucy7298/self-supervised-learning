from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from pytorch_lightning.core.datamodule import LightningDataModule
from models import model_utils 
from typing import Sequence, Optional
import os

class FolderDataset(LightningDataModule): 
    def __init__(self, 
                data_path: str, 
                preprocess_train: Sequence[model_utils.ModuleConfig],
                preprocess_val: Sequence[model_utils.ModuleConfig], 
                preprocess_test: Sequence[model_utils.ModuleConfig], 
                train_val_test_dir: Sequence[str],
                batch_size: int): 
        super().__init__()
        self.data_path = data_path
        self.train_transform = model_utils.build_augmentations(preprocess_train)
        self.val_transform = model_utils.build_augmentations(preprocess_val)
        self.test_transform = model_utils.build_augmentations(preprocess_test)
        self.batch_size = batch_size
        self.dirnames = train_val_test_dir

    def train_dataloader(self): 
        train_path = os.path.join(self.data_path, self.dirnames[0])
        train_dataset = ImageFolder(root=train_path, transform=self.train_transform)
        return DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, drop_last=True)

    def val_dataloader(self): 
        val_path = os.path.join(self.data_path, self.dirnames[1])
        val_dataset = ImageFolder(root=val_path, transform=self.val_transform)
        return DataLoader(val_dataset, batch_size=self.batch_size, shuffle=True, drop_last=False)

    def test_dataloader(self): 
        test_path = os.path.join(self.data_path, self.dirnames[2])
        test_dataset = ImageFolder(root=test_path, transform=self.test_transform)
        return DataLoader(test_dataset, batch_size=self.batch_size, shuffle=True, drop_last=False)
