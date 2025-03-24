import yaml
from typing import List
from ...utils import HyperParams
from dataclasses import dataclass, field



@dataclass
class VectorPromptHyperParams(HyperParams):
    # Method (with predefined values)
    alg_name: str = 'vector_prompt'
    save_vectors: bool = True
    layers: List[int] = field(default_factory=lambda: list(range(32)))
    steer_train_dataset: str = "safeedit"
    steer_vector_output_dir: str = "../"
    prompt: str = "Write a sentence about"
    generate_prompt_params: dict = field(default_factory=lambda: {})
    
    @classmethod
    def from_hparams(cls, hparams_name_or_path: str):

        if '.yaml' not in hparams_name_or_path:
            hparams_name_or_path = hparams_name_or_path + '.yaml'

        with open(hparams_name_or_path, "r") as stream:
            config = yaml.safe_load(stream)
            config = super().construct_float_from_scientific_notation(config)

        assert (config and config['alg_name'] == 'vector_prompt') or print(f'VectorPromptHyperParams can not load from {hparams_name_or_path}, '
                                                f'alg_name is {config["alg_name"]} ')
        return cls(**config)
