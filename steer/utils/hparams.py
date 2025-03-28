import json
from dataclasses import dataclass
from dataclasses import asdict


@dataclass
class HyperParams:
    """
    Simple wrapper to store hyperparameters for Python-based rewriting methods.
    """
    use_chat_template: bool=False
    system_prompt: str=''
    torch_dtype: str='fp32'
    seed: int=42
    model_name_or_path: str='your_own_path'
    device: str='cpu'
    use_cache: bool=True

    @classmethod
    def from_json(cls, fpath):
        with open(fpath, "r") as f:
            data = json.load(f)

        return cls(**data)

    def construct_float_from_scientific_notation(config: dict):
        for key, value in config.items():
            if isinstance(value, str):
                try:
                    # Convert scalar to float if it is in scientific notation format
                    config[key] = float(value)
                except:
                    pass
        return config

    def to_dict(config) -> dict:
        dict = asdict(config)
        return dict