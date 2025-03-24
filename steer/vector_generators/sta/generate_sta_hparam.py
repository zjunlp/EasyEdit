import yaml
from typing import List
from ...utils import HyperParams
from dataclasses import dataclass, field



@dataclass
class STAHyperParams(HyperParams):
    # Method (with predefined values)
    alg_name: str = 'sta'
    save_vectors: bool = True
    layers: List[int] = field(default_factory=lambda: [24])
    sae_paths: List[str] = field(default_factory=lambda: ['/layer_24/width_16k/average_l0_114'])
    trims:  List[float] = field(default_factory=lambda: [0.65])
    steer_train_dataset: str = "safeedit"
    mode: str = "act_and_freq"
    multiple_choice: bool = False
    steer_vector_output_dir: str = "../"


    @classmethod
    def from_hparams(cls, hparams_name_or_path: str):

        if '.yaml' not in hparams_name_or_path:
            hparams_name_or_path = hparams_name_or_path + '.yaml'

        with open(hparams_name_or_path, "r") as stream:
            config = yaml.safe_load(stream)
            config = super().construct_float_from_scientific_notation(config)

        assert (config and config['alg_name'] == 'sta') or print(f'STAHyperParams can not load from {hparams_name_or_path}, '
                                                f'alg_name is {config["alg_name"]} ')
        return cls(**config)
