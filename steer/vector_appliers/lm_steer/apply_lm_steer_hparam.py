from dataclasses import dataclass
from typing import Optional, List
import yaml
from ...utils import HyperParams

@dataclass
class ApplyLmSteerHyperParams(HyperParams):
    # Model related
    alg_name: str='caa'
    steer_vector_load_dir: Optional[str]='vectors'
    adaptor_class: str='multiply'
    adapted_component: str='final_layer'
    epsilon: float=0.001
    init_var: float=0.01
    rank: int=1000
    num_steers: int=2 
    steer_values: List[float]=None
    generate_orig_output: bool=False

    @classmethod
    def from_hparams(cls, hparams_name_or_path: str):

        if '.yaml' not in hparams_name_or_path:
            hparams_name_or_path = hparams_name_or_path + '.yaml'

        with open(hparams_name_or_path, "r") as stream:
            config = yaml.safe_load(stream)
            config = super().construct_float_from_scientific_notation(config)

        assert (config and config['alg_name'] == 'lm_steer') or print(f'ApplyLmSteerHyperParams can not load from {hparams_name_or_path}, '
                                                f'alg_name is {config["alg_name"]} ')
        return cls(**config)