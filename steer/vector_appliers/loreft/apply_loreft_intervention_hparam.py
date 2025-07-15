import yaml
from typing import List
from ...utils import HyperParams
from dataclasses import dataclass, field



@dataclass
class ApplyLoReFTHyperParams(HyperParams):
    # Method (with predefined values)
    alg_name: str = 'loreft'
    steer_vector_load_dir: str = None
    reft_layers: List[int] = field(default_factory=lambda: [20,21])
    max_length: int = 512
    batch_size: int = 1
    device: str = 'cuda'
    low_rank_dimension: int = 1
    position: str = "l1"
    temperature: float = 1.0


    @classmethod
    def from_hparams(cls, hparams_name_or_path: str):

        if '.yaml' not in hparams_name_or_path:
            hparams_name_or_path = hparams_name_or_path + '.yaml'

        with open(hparams_name_or_path, "r") as stream:
            config = yaml.safe_load(stream)
            config = super().construct_float_from_scientific_notation(config)

        assert (config and config['alg_name'] == 'loReFT') or print(f'LoReFTHyperParams can not load from {hparams_name_or_path}, '
                                                f'alg_name is {config["alg_name"]} ')
        return cls(**config)
