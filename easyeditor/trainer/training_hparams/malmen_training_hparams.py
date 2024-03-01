from dataclasses import dataclass
from ...util.hparams import HyperParams
from typing import Optional, Any, List
import yaml


@dataclass
class MALMENTrainingHparams(HyperParams):

    # Model
    model_name: str
    model_class: str
    tokenizer_class: str
    tokenizer_name: str
    inner_params: List[str]
    
    archive: Any

    # Method
    alg: str
    debug: bool
    dropout: float
    train_base: bool
    no_grad_layers: Any

    rank: int
    n_edits: int
    n_blocks: int
    lr: float
    meta_lr: float
    loc_coef: float
    max_grad_norm: float
    token: str

    # Output
    results_dir: str

    # Train
    device: str
    batch_size: int
    editor_batch_size: int
    silent: bool
    log_interval: int
    eval_log_interval:int
    final_eval:bool
    val_interval: int
    early_stop_patience: int
    early_stop_key: str
    eval_only: bool
    save: bool

    val_batch_size: Optional[int]
    val_steps: int

    model_save_pt: Optional[int]=5000
    half: Optional[bool] = False
    model_parallel: bool = False
    max_epochs: Optional[int] = None
    max_iters: Optional[int] = None

    @classmethod
    def from_hparams(cls, hparams_name_or_path: str):

        if '.yaml' not in hparams_name_or_path:
            hparams_name_or_path = hparams_name_or_path + '.yaml'

        with open(hparams_name_or_path, "r") as stream:
            config = yaml.safe_load(stream)
            config = super().construct_float_from_scientific_notation(config)

        assert (config and config['alg'] == 'MALMEN') or print(f'MALMENTrainingHyperParams can not load from {hparams_name_or_path}, '
                                                f'alg_name is {config["alg"]} ')
        config['val_batch_size'] = config['batch_size']
        return cls(**config)
    
