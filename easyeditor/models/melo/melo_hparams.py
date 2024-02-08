from dataclasses import dataclass
from typing import List
from ...util.hparams import HyperParams
import yaml

@dataclass
class GRACEHyperParams(HyperParams):
    name: str
    num_iter: int
    init_radius: float
    dist_fn: str # euc, mmd, cos
    val_init: str # cold, warm
    val_train: str # sgd, pert
    val_reg: bool # early
    reg: str # early_stop
    replacement: str # replace_last, replace_all, replace_prompt
    expand_mode: str # , moving_avg, decay
    num_pert: int # only matters when using perturbation training
    key_id: int
    num_edit_per_block: int
    num_block: int
    num_rank_per_block: int
    metric_period: int
    edit_lr: float

@dataclass
class MODELHyperParams(HyperParams):
    name: str
    class_name: str
    tokenizer_class: str
    tokenizer_name: str
    fan_in_fan_out: bool
    target_modules: list[str]
    pt: str # set this to 'hallucination' inside your checkpoint directory
    grace_layer: str
@dataclass
class LoRAHyperParams(HyperParams):
  cls_name: str
  cls_class: str
  supervised: bool
  cos: bool
  freeze: str
  square: bool
  bound_embeds: bool
  use_all_negatives: bool
  freeze_lora: bool
  dist_heads: int
  cross_attend: bool
  soft_weighting: bool
  checkpoint_grad: bool
  lora_r: int
  lora_alpha: int
  lora_dropout: float
 
@dataclass
class MELOHyperParams(HyperParams):
    alg_name: str
    model_parallel: bool
    device: int
    max_length: int
    task: str
    lora_task_type: str
    check_dir: str
    grace: GRACEHyperParams
    model: MODELHyperParams
    lora: LoRAHyperParams
    
    @classmethod
    def from_hparams(cls, hparams_name_or_path: str):
        if '.yaml' not in hparams_name_or_path:
            hparams_name_or_path = hparams_name_or_path + '.yaml'

        with open(hparams_name_or_path, "r") as stream:
            config = yaml.safe_load(stream)
            config = super().construct_float_from_scientific_notation(config)

        assert (config and config['alg_name'] == 'MELO') or print(
            f'GraceHyperParams can not load from {hparams_name_or_path}, '
            f'alg_name is {config["alg_name"]} ')
        
        grace_config = GRACEHyperParams(**config['grace'])
        config['grace'] = grace_config
        model_config = MODELHyperParams(**config['model'])
        config['model'] = model_config
        lora_config = LoRAHyperParams(**config['lora'])
        config['lora'] = lora_config
        return cls(**config)
