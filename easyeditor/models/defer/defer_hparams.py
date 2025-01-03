from dataclasses import dataclass
from typing import List
from ...util.hparams import HyperParams
import yaml


@dataclass
class DeferHyperParams(HyperParams):
    # Experiments
    
    edit_lr: int
    n_iter: int
    # Method
    threshold: float

    # Module templates
    inner_params: List[str]
    device: int
    alg_name: str
    model_name: str

    # Defaults
    batch_size: int = 1
    max_length: int = 30
    model_parallel: bool = False

    @classmethod
    def from_hparams(cls, hparams_name_or_path: str):
        if '.yaml' not in hparams_name_or_path:
            hparams_name_or_path = hparams_name_or_path + '.yaml'

        with open(hparams_name_or_path, "r") as stream:
            config = yaml.safe_load(stream)
            config = super().construct_float_from_scientific_notation(config)

        assert (config and config['alg_name'] == 'DEFER') or print(
            f'GraceHyperParams can not load from {hparams_name_or_path}, '
            f'alg_name is {config["alg_name"]} ')
        return cls(**config)
