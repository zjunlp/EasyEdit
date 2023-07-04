from dataclasses import dataclass
from ...util.hparams import HyperParams
from typing import Optional, Any, List
import yaml


@dataclass
class SERACTrainingHparams(HyperParams):

    model_name: str
    model_class: str
    small_name: str
    tokenizer_class: str
    tokenizer_name: str
    cls_name: str
    cls_class: str
    inner_params: List[str]

    archive: Any

    # Method
    alg: str
    lr: float
    edit_lr: float
    seed: int
    lr_lr: float
    cedit: float
    cloc: float
    cbase: float
    dropout: float
    final_eval: bool
    supervised: bool
    train_base: bool
    no_grad_layers: Any
    soft_weighting: bool
    checkpoint_grad: bool
    cross_attend: bool
    cos: bool
    freeze: Any
    square: bool
    bound_embeds: bool
    use_all_negatives: bool
    freeze_cntr: bool
    dist_heads: int
    lora: Any

    # Output
    results_dir: str

    # Train
    device: str
    batch_size: int
    model_save_pt: int
    edit_bs: int
    silent: bool
    log_interval: int
    val_interval: int
    early_stop_patience: int
    early_stop_key: str
    eval_only: bool
    half: bool
    save: bool
    debug: bool
    log_errors: bool
    unlikelihood: bool

    val_batch_size: int
    accumulate_bs: int
    val_steps: int
    opt: str
    grad_clip: float

    max_epochs: Optional[int] = None
    max_iters: Optional[int] = None
    max_length: int = 32


    @classmethod
    def from_hparams(cls, hparams_name_or_path: str):

        if '.yaml' not in hparams_name_or_path:
            hparams_name_or_path = hparams_name_or_path + '.yaml'

        with open(hparams_name_or_path, "r") as stream:
            config = yaml.safe_load(stream)
            config = super().construct_float_from_scientific_notation(config)


        assert (config and config['alg'] == 'SERAC') or print(f'SERACTrainingHyperParams can not load from {hparams_name_or_path}, '
                                                f'alg_name is {config["alg"]} ')
        return cls(**config)
