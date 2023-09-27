from dataclasses import dataclass
import yaml
from typing import Optional

from ...util.hparams import HyperParams


@dataclass
class FTApiHyperParams(HyperParams):
    api_key: str
    results_dir: str

    alg_name: str
    model_name: str
    proxy: Optional[str] = None

    @classmethod
    def from_hparams(cls, hparams_name_or_path: str):

        if '.yaml' not in hparams_name_or_path:
            hparams_name_or_path = hparams_name_or_path + '.yaml'

        with open(hparams_name_or_path, "r") as stream:
            config = yaml.safe_load(stream)
            config = super().construct_float_from_scientific_notation(config)

        assert (config and config['alg_name'] == 'FT-Api') or print(f'FTApiHyperParams can not load from {hparams_name_or_path}, '
                                                f'alg_name is {config["alg_name"]} ')
        return cls(**config)
