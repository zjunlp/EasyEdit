from dataclasses import dataclass
from typing import List
import yaml

from ...util.hparams import HyperParams


@dataclass
class DeCoHyperParams(HyperParams):
    device: int
    alg_name: str
    model_name: str
    model_parallel: bool
    alpha: float
    threshold_top_p: float
    threshold_top_k: int
    early_exit_layers: List[int]
    @classmethod
    def from_hparams(cls, hparams_name_or_path: str):

        if '.yaml' not in hparams_name_or_path:
            hparams_name_or_path = hparams_name_or_path + '.yaml'

        with open(hparams_name_or_path, "r") as stream:
            config = yaml.safe_load(stream)
            config = super().construct_float_from_scientific_notation(config)

        return cls(**config)
