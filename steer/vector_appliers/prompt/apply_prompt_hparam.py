from dataclasses import dataclass, field
from typing import Optional, List
import yaml
from ...utils import HyperParams

@dataclass
class ApplyPromptHyperParams(HyperParams):
    # Model related
    alg_name: str = 'prompt'
    model_name_or_path: str = 'EleutherAI/gpt-neo-2.7B'
    ckpt_name: Optional[str] = None
    device: str = 'cuda'
    
    # Chat template related
    system_message: Optional[str] = None  # System message
    chat_template: Optional[str] = None  # Custom chat template, use model default if None
    
    # Prompt related
    prompt: str = None  # User prompt
    generate_prompt_params: dict = field(default_factory=lambda: {})
    # Generate related
    data_path: Optional[str] = None
    generation_output_dir: Optional[str] = None
    generation_data_size: Optional[int] = None
    top_p: float = 0.9
    temperature: float = 0.7
    max_new_tokens: int = 100
    num_return_sequences: int = 1
    do_sample: bool = True
    
    # Evaluation related
    num_responses: int = 25  
    prompt_length: int = 20  
    generate_orig_output: bool=False
    
    @classmethod
    def from_hparams(cls, hparams_name_or_path: str):
        if '.yaml' not in hparams_name_or_path:
            hparams_name_or_path = hparams_name_or_path + '.yaml'

        with open(hparams_name_or_path, "r") as stream:
            config = yaml.safe_load(stream)
            config = super().construct_float_from_scientific_notation(config)

        assert (config and config['alg_name'] == 'prompt') or print(f'PromptHyperParams can not load from {hparams_name_or_path}, '
                                                f'alg_name is {config["alg_name"]} ')
        return cls(**config)