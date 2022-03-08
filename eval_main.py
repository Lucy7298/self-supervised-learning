import hydra
from omegaconf import DictConfig

@hydra.main(config_path="configs/", config_name="evaluate_config.yaml")
def main(config: DictConfig):

    # Imports should be nested inside @hydra.main to optimize tab completion
    # Read more here: https://github.com/facebookresearch/hydra/issues/934
    from awesome_ssl.evaluate import evaluate
    
    # Debug model
    return evaluate(config)


if __name__ == "__main__":
    main()