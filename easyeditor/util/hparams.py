import json
from dataclasses import dataclass
from dataclasses import asdict

import yaml


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
    
    def to_dict(config) -> dict:
        dict = asdict(config)
        return dict


def load_hparams_config(hparams_name_or_path: str) -> dict:
    hparams_path = str(hparams_name_or_path)
    if not hparams_path.endswith((".yaml", ".yml", ".json")):
        hparams_path = hparams_path + ".yaml"

    with open(hparams_path, "r") as stream:
        if hparams_path.endswith(".json"):
            config = json.load(stream)
        else:
            config = yaml.safe_load(stream)

    if config is None:
        return config
    if not isinstance(config, dict):
        raise ValueError(f"hparams config must be a mapping: {hparams_path}")
    return HyperParams.construct_float_from_scientific_notation(config)


def normalize_alg_name(
    config: dict,
    expected_alg_name: str,
    *,
    aliases: dict = None,
    legacy_key: str = "alg",
    keep_legacy: bool = False,
    legacy_value: str = None,
) -> dict:
    if not config:
        raise ValueError(f"{expected_alg_name} hparams config is empty.")

    aliases = aliases or {}

    def canonical(value):
        if value is None:
            return None
        value = str(value)
        return aliases.get(value, value)

    present = [
        (key, config.get(key), canonical(config.get(key)))
        for key in ("alg_name", legacy_key)
        if config.get(key) is not None
    ]
    if not present:
        raise ValueError(
            f"{expected_alg_name} hparams must define alg_name or {legacy_key}."
        )

    invalid = [
        f"{key}={value!r}"
        for key, value, name in present
        if name != expected_alg_name
    ]
    if invalid:
        details = ", ".join(invalid)
        raise ValueError(f"{expected_alg_name} hparams can not load with {details}.")

    config["alg_name"] = expected_alg_name
    if keep_legacy:
        if legacy_key not in config or config.get(legacy_key) is None:
            config[legacy_key] = legacy_value or expected_alg_name
    else:
        config.pop(legacy_key, None)
    return config
