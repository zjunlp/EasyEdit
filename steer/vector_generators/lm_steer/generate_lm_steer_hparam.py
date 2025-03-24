import yaml
from dataclasses import dataclass
from ...utils import HyperParams



@dataclass
class LmSteerHyperParams(HyperParams):
    
    alg_name: str = 'lm_steer'

    steer_vector_output_dir: str=None
    subset: int=None
    dummy_steer: int=1
    
    steer_train_dataset: str=None
    regularization: float = 0
    optimizer: str = "Adam"
    lr: float = 0.001
    gamma_mean: float = 0.99
    n_steps: int = 10000
    max_length: int = 256
    batch_size: int = 32
    log_step: int = 500
    training_steer: int = 0

    adaptor_class: str = "multiply"
    adapted_component: str = "final_layer"
    epsilon: float = 0.001
    init_var: float = 0.01
    rank: int = 1000
    num_steers: int = 10
    low_resource_mode: bool = False
    save_vectors: bool = True

    @classmethod
    def from_hparams(cls, hparams_name_or_path: str):

        if '.yaml' not in hparams_name_or_path:
            hparams_name_or_path = hparams_name_or_path + '.yaml'

        with open(hparams_name_or_path, "r") as stream:
            config = yaml.safe_load(stream)
            config = super().construct_float_from_scientific_notation(config)

        assert (config and config['alg_name'] == 'lm_steer') or print(f'LmSteerHyperParams can not load from {hparams_name_or_path}, '
                                                f'alg_name is {config["alg_name"]} ')
        return cls(**config)
