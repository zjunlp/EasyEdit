from dataclasses import dataclass, field
from typing import Optional, List
import yaml
from ...utils.hparams import HyperParams

@dataclass
class ApplyPromptHyperParams(HyperParams):
    # Model related
    alg_name: str = 'prompt'
    # Prompt related
    prompt: str = None  # User prompt
    generate_prompt_params: dict = field(default_factory=lambda: {})

    
    @classmethod
    def from_hparams(cls, hparams_name_or_path: str):
        if '.yaml' not in hparams_name_or_path:
            hparams_name_or_path = hparams_name_or_path + '.yaml'

        with open(hparams_name_or_path, "r") as stream:
            config = yaml.safe_load(stream)
            config = super().construct_float_from_scientific_notation(config)

        assert (config and config['alg_name'] == 'prompt') or print(f'PromptHyperParams can not load from {hparams_name_or_path}, '
                                                f'alg_name is {config["alg_name"]} ')
        return cls(**config)