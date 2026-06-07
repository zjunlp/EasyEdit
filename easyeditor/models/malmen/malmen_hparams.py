from dataclasses import dataclass
from ...util.hparams import HyperParams, load_hparams_config, normalize_alg_name
from typing import Optional, Any, List


@dataclass
class MALMENHyperParams(HyperParams):
    alg_name: str
    
    # Model
    model_name: str
    model_class: str
    tokenizer_class: str
    tokenizer_name: str
    inner_params: List[str]
    device: int
    archive: Any

    # Method
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

    max_length: int = 40

    model_save_pt: Optional[int]=5000
    half: Optional[bool] = False
    model_parallel: bool = False
    max_epochs: Optional[int] = None
    max_iters: Optional[int] = None

    @classmethod
    def from_hparams(cls, hparams_name_or_path: str):

        config = load_hparams_config(hparams_name_or_path)
        config = normalize_alg_name(config, "MALMEN")
        config['val_batch_size'] = config['batch_size']
        return cls(**config)
    
