import hydra
from omegaconf import DictConfig

@hydra.main(config_path="configs/", config_name="config.yaml")
def main(config: DictConfig):

    # Imports should be nested inside @hydra.main to optimize tab completion
    # Read more here: https://github.com/facebookresearch/hydra/issues/934
    from awesome_ssl.train import train
    
    # Train model
    return train(config)


if __name__ == "__main__":
    main()