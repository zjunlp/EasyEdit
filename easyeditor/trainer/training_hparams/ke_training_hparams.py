from dataclasses import dataclass
from ...util.hparams import HyperParams
from typing import Optional, Any, List
import yaml


@dataclass
class KETrainingHparams(HyperParams):

    # Model
    model_name: str
    model_class: str
    tokenizer_class: str
    tokenizer_name: str
    inner_params: List[str]

    archive: Any

    # Method
    alg: str
    lr: float
    edit_lr: float
    lr_lr: float
    seed: int
    debug: bool
    cedit: float
    cloc: float
    cbase: float
    dropout: float
    train_base: bool
    no_grad_layers: Any

    # Output

    results_dir: str


    # Train
    device: str
    batch_size: int
    model_save_pt: int
    silent: bool
    log_interval: int
    eval_log_interval:int
    final_eval:bool
    val_interval: int
    early_stop_patience: int
    early_stop_key: str
    eval_only: bool
    half: bool
    save: bool
    verbose: bool

    val_batch_size: int
    accumulate_bs: int
    val_steps: int
    opt: str
    grad_clip: float

    max_epochs: Optional[int] = None
    max_iters: Optional[int] = None


    @classmethod
    def from_hparams(cls, hparams_name_or_path: str):

        if '.yaml' not in hparams_name_or_path:
            hparams_name_or_path = hparams_name_or_path + '.yaml'

        with open(hparams_name_or_path, "r") as stream:
            config = yaml.safe_load(stream)
            config = super().construct_float_from_scientific_notation(config)

        assert (config and config['alg'] == 'KE') or print(f'KETrainingHyperParams can not load from {hparams_name_or_path}, '
                                                f'alg_name is {config["alg"]} ')
        return cls(**config)

