from dataclasses import dataclass

from ...util.hparams import HyperParams, load_hparams_config, normalize_alg_name
from typing import Optional, Any, List


@dataclass
class MENDHyperParams(HyperParams):
    model_class: str
    tokenizer_class: str
    tokenizer_name: str
    inner_params: List[str]

    archive: Any

    # Method
    lr: float
    edit_lr: float
    lr_lr: float
    lr_scale: float
    seed: int
    debug: bool
    cedit: float
    cloc: float
    cbase: float
    dropout: float
    train_base: bool
    no_grad_layers: Any
    one_sided: bool
    n_hidden: int
    hidden_dim: Any
    init: str
    norm: bool
    combine: bool
    x_only: bool
    delta_only: bool
    act: str
    rank: int
    mlp_class: str
    shared: bool

    # Output
    results_dir: str

    # Train
    device: int
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

    alg_name: str
    model_name: str
    device: int

    batch_size: int = 1
    max_length: int = 40
    max_output_length: Optional[int] = None
    max_epochs: Optional[int] = None
    max_iters: Optional[int] = None

    model_parallel: bool = False

    @classmethod
    def from_hparams(cls, hparams_name_or_path: str):

        config = load_hparams_config(hparams_name_or_path)
        config = normalize_alg_name(config, "MEND")
        return cls(**config)
