from slot_based_disentanglement.utils import hydra_custom_resolvers
import hydra
from omegaconf import DictConfig

# python run_training.py training=<evaluation_config> ckpt_path=<path_to_ckpt_to_evaluate> run_name=<run_name>


@hydra.main(config_path="configs", config_name="train_root")
def main(hydra_config: DictConfig):
    # Imports should be nested inside @hydra.main to optimize tab completion
    # Read more here: https://github.com/facebookresearch/hydra/issues/934

    import slot_based_disentanglement.utils.general as utils
    from slot_based_disentanglement.training_pipeline import train

    # Applies optional utilities:
    # - disabling python warnings
    # - prints config
    utils.extras(hydra_config)

    # Train model
    train(hydra_config)


if __name__ == "__main__":
    main()
