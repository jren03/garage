import hydra
import omegaconf


from garage.algorithms import model_based_irl, model_free_irl, pretrain_antmaze, bc
from garage.utils.common import set_seed
from garage.utils.fetch_demos import fetch_demos

import numpy as np

# due to version > 1.20.0, np.bool_ is deprecated
np.bool = np.bool_


@hydra.main(
    config_path="config",
    config_name="main",
)
def main(cfg: omegaconf.DictConfig):
    set_seed(cfg.seed)
    demos_dict = fetch_demos(cfg.overrides.env, cfg.overrides.expert_dataset_size)

    if cfg.algorithm.name == "hyper":
        model_based_irl.train(cfg, demos_dict)
    elif cfg.algorithm.name == "pretrain_antmaze":
        pretrain_antmaze.train(cfg, demos_dict)
    elif cfg.algorithm.name == "bc":
        bc.train(cfg, demos_dict)
    else:
        model_free_irl.train(cfg, demos_dict)


if __name__ == "__main__":
    main()
