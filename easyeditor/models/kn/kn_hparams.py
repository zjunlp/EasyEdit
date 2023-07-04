from dataclasses import dataclass

from ...util.hparams import HyperParams
import yaml


@dataclass
class KNHyperParams(HyperParams):
    lr_scale: float
    n_toks: int
    model_name: str
    refine: bool
    batch_size: int
    steps: int
    adaptive_threshold: float
    p: float
    device: int
    alg_name: str

    max_length: int = 30
    @classmethod
    def from_hparams(cls, hparams_name_or_path: str):

        if '.yaml' not in hparams_name_or_path:
            hparams_name_or_path = hparams_name_or_path + '.yaml'

        with open(hparams_name_or_path, "r") as stream:
            config = yaml.safe_load(stream)
            config = super().construct_float_from_scientific_notation(config)

        assert (config and config['alg_name'] == 'KN') or print(f'KNHyperParams can not load from {hparams_name_or_path}, '
                                                f'alg_name is {config["alg_name"]} ')
        return cls(**config)
