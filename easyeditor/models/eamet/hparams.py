from dataclasses import dataclass
from typing import List, Literal, Union

from ...util.hparams import HyperParams, load_hparams_config, normalize_alg_name


@dataclass
class EAMETHyperParams(HyperParams):
    # Method
    layers: List[int]
    layer_selection: Literal["all", "random"]
    fact_token: Literal[
        "last", "subject_first", "subject_last", "subject_first_after_last"
    ]
    v_num_grad_steps: int
    v_lr: float
    ks_bs: int
    v_loss_layer: int
    v_weight_decay: float
    clamp_norm_factor: float
    clamp_ks_factor: float
    kl_factor: float
    cs_factor: float
    mse_factor: float
    tau: float
    norm_factor: float

    delta_init: Literal["target_init", "zero"]
    weight_decay_method: Literal["norm", "fix_ks"]

    mom2_adjustment: bool
    mom2_update_weight: List[float]

    # Module templates
    rewrite_module_tmp: str
    layer_module_tmp: str
    mlp_module_tmp: str
    attn_module_tmp: str
    ln_f_module: str
    lm_head_module: str

    # Statistics
    mom2_dataset: str
    mom2_n_samples: int
    mom2_dtype: str
    alg_name: str
    device: Union[int, str]
    model_name: str
    stats_dir: str

    max_length: int = 40
    batch_size: int = 1
    model_parallel: bool = False

    @classmethod
    def from_hparams(cls, hparams_name_or_path: str):
        config = load_hparams_config(hparams_name_or_path)
        config = normalize_alg_name(config, "EAMET")
        if not isinstance(config["mom2_update_weight"], list):
            config["mom2_update_weight"] = [
                config["mom2_update_weight"]
                for _ in config["layers"]
            ]
        return cls(**config)
