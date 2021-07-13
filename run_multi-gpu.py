"""

python "/media/data/jacob/GitHub/lightning-hydra-classifiers/run_multi-gpu.py"

python "/media/data/jacob/GitHub/lightning-hydra-classifiers/run_basic.py" +experiment=1_pnas_exp_example_full

"""


import dotenv
import hydra
from omegaconf import DictConfig, OmegaConf

# load environment variables from `.env` file if it exists
# recursively searches for `.env` in all folders starting from work dir
dotenv.load_dotenv(override=True)

@hydra.main(config_path="configs/", config_name="multi-gpu")
# @hydra.main(config_path="configs/", config_name="PNAS_config") # ="config")
def main(config: DictConfig):

    # Imports should be nested inside @hydra.main to optimize tab completion
    # Read more here: https://github.com/facebookresearch/hydra/issues/934
    from lightning_hydra_classifiers.train_multi_gpu import train
    from lightning_hydra_classifiers.utils import template_utils
    
    # A couple of optional utilities:
    # - disabling python warnings
    # - easier access to debug mode
    # - forcing debug friendly configuration
    # - forcing multi-gpu friendly configuration
    # You can safely get rid of this line if you don't want those
    template_utils.extras(config)
    OmegaConf.set_struct(config, False)

    # Pretty print config using Rich library
    if config.get("print_config"):
        template_utils.print_config(config, resolve=True)

#     return run_full_tuned_experiment(config)
    return train(config)


if __name__ == "__main__":
    main()
