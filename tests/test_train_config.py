from hydra.utils import instantiate 
from hydra import initialize, compose
import pytest
import os
import torch

MY_PATH = os.path.abspath(os.path.dirname(__file__))

def get_config_root():
    abs_path = get_abs_path("../configs")
    return os.path.relpath(abs_path, start=MY_PATH)

def get_abs_path(relative_path): 
    return os.path.normpath(os.path.join(MY_PATH, relative_path))

@pytest.mark.parametrize(
    "dataset_path", 
    os.listdir(get_abs_path("../configs/val_dataset"))
)
def test_dataloader_configs(dataset_path): 
    with initialize(get_config_root()): 
        dataset_path = os.path.join("val_dataset", dataset_path)
        dataset = instantiate(compose(dataset_path).val_dataset)
        # test that it's a working dataset 
        im, label = dataset[0]
        assert torch.is_tensor(im)

def test_train_dataloader_config(): 
    with initialize(get_config_root()): 
        dataset = instantiate(compose("train_dataset/imagenet_byol.yaml").train_dataset)
        im1, im2, label = dataset[0]
        assert torch.is_tensor(im1)
        assert torch.is_tensor(im2)
    