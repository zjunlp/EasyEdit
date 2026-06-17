from dataclasses import dataclass
from ...util.hparams import HyperParams, load_hparams_config, normalize_alg_name
from typing import Optional, Any, List


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
    dropout: float
    no_grad_layers: Any
    batch_size: int
    lr: float
    token: str
    batch_size_once: int
    editor_batch_size: int
    silent: bool
    # Output
    results_dir: str
    max_length: int = 40

    half: Optional[bool] = False
    model_parallel: bool = False



    @classmethod
    def from_hparams(cls, hparams_name_or_path: str):

        config = load_hparams_config(hparams_name_or_path)
        config = normalize_alg_name(
            config,
            "ULTRAEDIT",
            aliases={"UltraEdit": "ULTRAEDIT"},
        )
        class_name = config.pop("class_name", None)
        if class_name is not None:
            if "model_class" not in config:
                config["model_class"] = class_name
            elif config["model_class"] != class_name:
                raise ValueError(
                    "ULTRAEDIT hparams define conflicting model_class and class_name."
                )
        return cls(**config)
    
