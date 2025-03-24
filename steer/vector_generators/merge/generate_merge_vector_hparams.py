import yaml
from dataclasses import dataclass
from typing import List
from ...utils import HyperParams


@dataclass
class MergeVectorHyperParams(HyperParams):
    # Method
    alg_name: str = 'merge_vector'
    vector_paths: List[str] = None  
    steer_vector_output_dir: str = None
    
    method: str = 'ties'
    weights: List[float] = None      
    densities: List[float] = None

    @classmethod
    def from_hparams(cls, hparams_name_or_path: str):
        if '.yaml' not in hparams_name_or_path:
            hparams_name_or_path = hparams_name_or_path + '.yaml'

        with open(hparams_name_or_path, "r") as stream:
            config = yaml.safe_load(stream)
            config = super().construct_float_from_scientific_notation(config)

        assert (config and config['alg_name'] == 'merge_vector') or print(f'MergeVectorHyperParams can not load from {hparams_name_or_path}, '
                                                f'alg_name is {config["alg_name"]} ')
        return cls(**config)
