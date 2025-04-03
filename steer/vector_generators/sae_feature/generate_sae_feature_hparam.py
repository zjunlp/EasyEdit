import yaml
from typing import List
from ...utils import HyperParams
from dataclasses import dataclass, field



@dataclass
class SaeFeatureHyperParams(HyperParams):
    # Method (with predefined values)
    alg_name: str = 'sae_feature'
    layer: int = 24
    sae_path: str = None
    save_vectors: bool = True
    release: str = 'gemma-scope-9b-pt-res-canonical'
    sae_id: str = 'layer_24/width_16k/canonical'
    position_ids:  List[int] = field(default_factory=lambda: [0])
    strengths:  List[float] = field(default_factory=lambda: [1.5])
    steer_vector_output_dir: str = "../"


    @classmethod
    def from_hparams(cls, hparams_name_or_path: str):

        if '.yaml' not in hparams_name_or_path:
            hparams_name_or_path = hparams_name_or_path + '.yaml'

        with open(hparams_name_or_path, "r") as stream:
            config = yaml.safe_load(stream)
            config = super().construct_float_from_scientific_notation(config)

        assert (config and config['alg_name'] == 'sae_feature') or print(f'SaeFeatureHyperParams can not load from {hparams_name_or_path}, '
                                                f'alg_name is {config["alg_name"]} ')
        return cls(**config)
