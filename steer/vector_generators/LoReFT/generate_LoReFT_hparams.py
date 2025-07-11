import yaml
from typing import List
from ...utils import HyperParams
from dataclasses import dataclass, field
import torch


@dataclass
class LoReFTHyperParams(HyperParams):
    # Method (with predefined values)
    alg_name: str = 'loreft'
    steer_vector_output_dir: str = None
    steer_train_dataset: str=None
    reft_layers: List[int] = field(default_factory=lambda: [20,21])
    lr: float = 0.0001
    n_epochs: int = 3
    max_length: int = 256
    batch_size: int = 3
    gradient_accumulation_steps: int = 1
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    seed: int = 42
    subset: int = None
    low_rank_dimension: int = 1
    position: str = "l1"
    weight_decay: float = 0.01
    save_vectors: bool = True
    use_cache : bool = True


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
