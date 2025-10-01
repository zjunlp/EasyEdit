from dataclasses import dataclass, field
from typing import Optional, List
import yaml

from ...utils import HyperParams

@dataclass
class ApplyRepsHyperParams(HyperParams):
    # Model related
    alg_name: str = 'reps_vector'
    layers: List[int] = field(default_factory=lambda: [24])
    multipliers: List[float] = field(default_factory=lambda: [1.0])
    
    # Vector loading related
    steer_vector_load_dir: str = None  # Base directory for vectors
    concept_id: int = 0  # Single concept
    
    @classmethod
    def from_hparams(cls, hparams_name_or_path: str):

        if '.yaml' not in hparams_name_or_path:
            hparams_name_or_path = hparams_name_or_path + '.yaml'

        with open(hparams_name_or_path, "r") as stream:
            config = yaml.safe_load(stream)
            config = super().construct_float_from_scientific_notation(config)

        assert (config and config['alg_name'] == 'reps') or print(f'ApplyRepsHyperParams can not load from {hparams_name_or_path}, '
                                                f'alg_name is {config["alg_name"]} ')
        return cls(**config)