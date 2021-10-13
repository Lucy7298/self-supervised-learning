# test that byol can build itself
# sample config is in configs/byol.yaml
import awesome_ssl
from awesome_ssl.models.model_utils import build_module
from hydra import initialize, compose
import pytest 
import torch

@pytest.fixture
def model_config(): 
    with initialize(config_path="../../configs/model"):
        # config is relative to a module
        return compose(config_name="byol")

def test_build_ssl(model_config) -> None:
    model = build_module(model_config)
    assert isinstance(model, awesome_ssl.models.byol.BYOL)

def test_training_step(model_config, mocker) -> None: 
    model = build_module(model_config) 
    mocker.patch.object(model, "optimizers", return_value=model.configure_optimizers()['optimizer'])
    fake_image = torch.rand(100, 3, 256, 256)
    out = model.training_step((fake_image, 0), 0)
