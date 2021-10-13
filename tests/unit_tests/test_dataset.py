from awesome_ssl.models.model_utils import build_module
from hydra import initialize, compose
import pytest 
import torch
from pytorch_lightning import LightningDataModule

DUMMY_BATCH_SIZE = 10

@pytest.fixture
def model_config(request): 
    with initialize(config_path="../../configs/dataset"):
        # config is relative to a module
        config = compose(config_name=request.param)
        config.kwargs.batch_size = DUMMY_BATCH_SIZE # make tests run faster
        print(config)
        return config

@pytest.mark.parametrize(
    'model_config',
    ('imagenet_byol.yaml',),
    indirect=True
)
def test_build(model_config) -> None:
    dataset = build_module(model_config)
    assert isinstance(dataset, LightningDataModule)

@pytest.mark.parametrize(
    'model_config',
    ('imagenet_byol.yaml',),
    indirect=True
)
def test_train_dataloader(model_config) -> None: 
    dataset : LightningDataModule = build_module(model_config)
    train_loader = dataset.train_dataloader()
    X, y = next(train_loader.__iter__())
    assert isinstance(X, torch.Tensor)
    assert isinstance(y, torch.Tensor)
    assert X.shape[0] == DUMMY_BATCH_SIZE

@pytest.mark.parametrize(
    'model_config',
    ('imagenet_byol.yaml',),
    indirect=True
)
def test_val_dataloader(model_config) -> None: 
    dataset : LightningDataModule = build_module(model_config)
    val_loader = dataset.val_dataloader()
    X, y = next(val_loader.__iter__())
    assert isinstance(X, torch.Tensor)
    assert isinstance(y, torch.Tensor)
    assert X.shape[0] == DUMMY_BATCH_SIZE

@pytest.mark.parametrize(
    'model_config',
    ('imagenet_byol.yaml',),
    indirect=True
)
def test_test_dataloader(model_config) -> None: 
    dataset : LightningDataModule = build_module(model_config)
    val_loader = dataset.test_dataloader()
    X, y = next(val_loader.__iter__())
    assert isinstance(X, torch.Tensor)
    assert isinstance(y, torch.Tensor)
    assert X.shape[0] == DUMMY_BATCH_SIZE
    