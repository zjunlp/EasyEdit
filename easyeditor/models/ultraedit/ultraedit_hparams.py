from dataclasses import dataclass
from ...util.hparams import HyperParams
from typing import Optional, Any, List
import yaml


@dataclass
class UltraEditHyperParams(HyperParams):
    alg_name: str
    
    # Model
    model_name: str
    model_class: str
    tokenizer_class: str
    tokenizer_name: str
    inner_params: List[str]
    device: int

    # Method
    alg: str
    dropout: float
    no_grad_layers: Any
    batch_size: int
    lr: float
    token: str
    batch_size_once: int
    editor_batch_size: int
    silent: bool
    max_length: int = 40

    half: Optional[bool] = False
    model_parallel: bool = False


    # Output
    results_dir: str

    @classmethod
    def from_hparams(cls, hparams_name_or_path: str):

        if '.yaml' not in hparams_name_or_path:
            hparams_name_or_path = hparams_name_or_path + '.yaml'

        with open(hparams_name_or_path, "r") as stream:
            config = yaml.safe_load(stream)
            config = super().construct_float_from_scientific_notation(config)

        assert (config and config['alg_name'] == 'ULTRAEDIT') or print(f'ULTRAEDITTrainingHyperParams can not load from {hparams_name_or_path}, '
                                                f'alg_name is {config["alg_name"]} ')
        return cls(**config)
    
