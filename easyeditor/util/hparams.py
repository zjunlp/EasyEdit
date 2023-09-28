import json
from dataclasses import dataclass


@dataclass
class HyperParams:
    """
    Simple wrapper to store hyperparameters for Python-based rewriting methods.
    """

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

    # @classmethod
    # def from_hparams(cls, hparams_name_or_path: str):
    #
    #     if '.yaml' not in hparams_name_or_path:
    #         hparams_name_or_path = hparams_name_or_path + '.yaml'
    #     config = compose(hparams_name_or_path)
    #
    #     assert config.alg_name in ALG_DICT.keys() or print(f'Editing Alg name {config.alg_name} not supported yet.')
    #
    #     params_class, apply_algo = ALG_DICT[config.alg_name]
    #
    #     return params_class(**config)
