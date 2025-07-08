import yaml
from typing import List
from ...utils import HyperParams
from dataclasses import dataclass, field

@dataclass
class RePSHyperParams(HyperParams):
    # === Basic Config ===
    alg_name: str = 'reps_vector'
    layers: List[int] = field(default_factory=lambda: list(range(32)))
    save_vectors: bool = True
    steer_vector_output_dir: str = "../"
    
    # === Dataset Config ===
    exclude_bos: bool = True
    max_concepts: int = 500
    max_num_of_examples: int = 10000
    steer_train_dataset: str = "safeedit"
    preference_pairs: List[str] = field(default_factory=lambda: ["orig_add", "orig_sub"]) 
    
    # === Training Config ===
    batch_size: int = 6  # the actual batch size also needs to multiply with |preference_pairs|
    beta: float = 1.0
    dropout: float = 0.1
    gemma: float = 0.0
    gradient_accumulation_steps: int = 1
    label_smoothing: float = 0.0
    loss_type: str = "scaled_simpo"
    lr: float = 0.08
    n_epochs: int = 18
    simpo_scaler: float = 1.0
    weight_decay: float = 0.00
    reference_free: bool = False
    train_on_negative: bool = True
    
    # === Intervention Config ===
    intervention_positions: str = "all"
    intervention_positions_dropout: float = 0.0
    intervention_type: str = "addition"  # clamping
    low_rank_dimension: int = 1
    
    # === Steering Config ===
<<<<<<< HEAD
    steering_factors: List[float] = field(default_factory=lambda: [0.0, 2.0])  
=======
    steering_factors: List[float] = field(default_factory=lambda: [2.0])  
>>>>>>> 2a46a1c5 (Forward Still Bug)
    steering_prompt_type: str = "blend_in"
    substraction_type: str = "null_it_out"  # normal or null_it_out

    @classmethod
    def from_hparams(cls, hparams_name_or_path: str):

        if '.yaml' not in hparams_name_or_path:
            hparams_name_or_path = hparams_name_or_path + '.yaml'

        with open(hparams_name_or_path, "r") as stream:
            config = yaml.safe_load(stream)
            config = super().construct_float_from_scientific_notation(config)

        assert (config and config['alg_name'] == 'reps') or print(f'RePSHyperParams can not load from {hparams_name_or_path}, ' f'alg_name is {config["alg_name"]} ')
        return cls(**config)
