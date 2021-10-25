import setuptools

setuptools.setup(
    name='awesome_ssl',
    version='0.1',
    install_requires=["torch", 
        "pytorch-lightning", 
        "omegaconf", 
        "kornia", 
        "hydra-core", 
        "lightning-bolts", 
        "torchvision", 
        "pytest", 
        "pytest-mock", 
        "wandb"],
    author_email='',
    packages=setuptools.find_packages(),
    zip_safe=False
)