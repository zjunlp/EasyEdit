from dataclasses import dataclass, field
from typing import List
from ...util.hparams import HyperParams
import yaml

@dataclass
class QLoRAHyperParams(HyperParams):
    # Model
    model_name: str
    # QLoRA specific
    quantization_bit: int = 4
    double_quant: bool = True
    quant_type: str = "nf4"
    # LoRA
    lora_type: str = "lora"
    lora_r: int = 8
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    target_modules: List[str] = field(default_factory=lambda: ["q_proj", "v_proj"])
    # Training
    num_steps: int = 100
    lr: float = 1e-4
    batch_size: int = 1
    max_length: int = 512
    weight_decay: float = 0.0
    # Device
    device: int = 1
    # Algorithm
    alg_name: str = "QLoRA"
    # Additional settings
    model_parallel: bool = False

    @classmethod
    def from_hparams(cls, hparams_name_or_path: str):
        if '.yaml' not in hparams_name_or_path:
            hparams_name_or_path = hparams_name_or_path + '.yaml'

        with open(hparams_name_or_path, "r") as stream:
            config = yaml.safe_load(stream)
            config = super().construct_float_from_scientific_notation(config)

        assert (config and config['alg_name'] == 'QLoRA') or print(
            f'QLoRAHyperParams can not load from {hparams_name_or_path}, '
            f'alg_name is {config["alg_name"]} ')
        return cls(**config)