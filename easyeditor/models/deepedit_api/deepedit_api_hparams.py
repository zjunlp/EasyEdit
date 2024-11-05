from dataclasses import dataclass
import yaml
from typing import Optional
from ...util.hparams import HyperParams

@dataclass
class DeepEditApiHyperParams(HyperParams):
    api_key: str
    results_dir: str

    prompts_dir: str
    contriver_dir: str
    tokenizer_dir: str
    alg_name: str
    model_name: str 
    device: int
    proxy: Optional[str] = None
    model_parallel = None
   

    @classmethod
    def from_hparams(cls, hparams_name_or_path: str):

        if '.yaml' not in hparams_name_or_path:
            hparams_name_or_path = hparams_name_or_path + '.yaml'

        with open(hparams_name_or_path, "r") as stream:
            config = yaml.safe_load(stream)
            config = super().construct_float_from_scientific_notation(config)

        assert (config and config['alg_name'] == 'DeepEdit-Api') or print(f'DeepEditApiHyperParams can not load from {hparams_name_or_path}, '
                                                f'alg_name is {config["alg_name"]} ')
        return cls(**config)