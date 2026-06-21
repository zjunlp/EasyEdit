import os
from omegaconf import OmegaConf
from .alg_dict import HYPERPARAMS_CLASS_DICT


def _oc_to_dict(cfg):
    """Convert OmegaConf/DictConfig (or plain dict) to a plain python dict."""
    if cfg is None:
        return {}
    try:
        return OmegaConf.to_container(cfg, resolve=True)  # type: ignore[arg-type]
    except Exception:
        return dict(cfg)


def _resolve_phase_hparams(method_hparams, phase: str) -> dict:
    """
    Resolve a single method hyperparam file into a flat dict for the given phase.

    Two supported file layouts:

    1) Merged file (recommended): shared fields at top level, plus `generate:` /
       `apply:` sections that only contain phase-specific fields.

           alg_name: caa
           layers: [20]
           generate:
             multiple_choice: false
           apply:
             multipliers: [1.0]

       For phase="generate" we merge: shared + generate section.
       For phase="apply"    we merge: shared + apply section.
       (phase section overrides shared on key conflicts)

    2) Flat file (legacy): no `generate:` / `apply:` sections, every field is at
       the top level. Returned as-is. This keeps all existing yaml files working.
    """
    method_hparams = _oc_to_dict(method_hparams)
    has_phase_sections = ("generate" in method_hparams) or ("apply" in method_hparams)
    if not has_phase_sections:
        return method_hparams

    shared = {k: v for k, v in method_hparams.items() if k not in ("generate", "apply")}
    phase_cfg = method_hparams.get(phase, {}) or {}
    if not isinstance(phase_cfg, dict):
        raise TypeError(f"'{phase}' section in hparam file must be a mapping, got {type(phase_cfg)}")
    return {**shared, **phase_cfg}


def load_generate_vector_hparams(top_cfg):
    """
    Load generate (train) hyperparams from the path list `steer_train_hparam_paths`.

    Each path may point to a merged method file (with `generate:` / `apply:`
    sections) or a legacy flat file; both are handled transparently.
    """
    hparams_dict = {}

    # Allow generate-only / apply-only configs without crashing.
    if not getattr(top_cfg, "steer_train_hparam_paths", None):
        return hparams_dict

    if isinstance(getattr(top_cfg, "steer_vector_output_dirs", None), str):
        top_cfg.steer_vector_output_dirs = [top_cfg.steer_vector_output_dirs]
    output_dirs = top_cfg.steer_vector_output_dirs

    for i, hparam_path in enumerate(top_cfg.steer_train_hparam_paths):
        assert os.path.exists(hparam_path), f"Hparam path {hparam_path} does not exist !"
        raw = OmegaConf.load(hparam_path)
        method_hparams = _resolve_phase_hparams(raw, phase="generate")
        print(hparam_path)

        alg_name = method_hparams["alg_name"]
        hparams_dict_key = f"{alg_name}"
        combined_hparams = {**method_hparams, **_oc_to_dict(top_cfg)}
        selected_hparams_class = HYPERPARAMS_CLASS_DICT[alg_name]["train"]
        intersect_keys = set(selected_hparams_class.__dataclass_fields__) & set(combined_hparams.keys())
        # remove extra fields
        hparams = selected_hparams_class(**{k: combined_hparams[k] for k in intersect_keys})
        print(f"Loading {alg_name} hparams from {hparam_path} ...")

        hparams.steer_vector_output_dir = (
            output_dirs[i] if i < len(output_dirs) else output_dirs[0]
        )
        hparams.steer_train_dataset = top_cfg.steer_train_dataset
        hparams_dict[hparams_dict_key] = hparams

    return hparams_dict


def load_apply_vector_hparams(top_cfg):
    """
    Load apply hyperparams from the path list `apply_steer_hparam_paths`.

    Each path may point to a merged method file (with `generate:` / `apply:`
    sections) or a legacy flat file; both are handled transparently.
    """
    hparams_dict = {}

    # if apply_steer_hparam_paths is not set, return empty dict
    if not getattr(top_cfg, "apply_steer_hparam_paths", None):
        return hparams_dict

    load_dirs = getattr(top_cfg, "steer_vector_load_dir", None)
    if isinstance(load_dirs, str):
        load_dirs = [load_dirs]

    for i, hparam_path in enumerate(top_cfg.apply_steer_hparam_paths):
        assert os.path.exists(hparam_path), f"Hparam path {hparam_path} does not exist !"
        raw = OmegaConf.load(hparam_path)
        method_hparams = _resolve_phase_hparams(raw, phase="apply")

        alg_name = method_hparams["alg_name"]
        combined_hparams = {**method_hparams, **_oc_to_dict(top_cfg)}
        selected_hparams_class = HYPERPARAMS_CLASS_DICT[alg_name]["apply"]
        intersect_keys = set(selected_hparams_class.__dataclass_fields__) & set(combined_hparams.keys())
        # remove extra fields
        hparams = selected_hparams_class(**{k: combined_hparams[k] for k in intersect_keys})

        if load_dirs is not None:
            hparams.steer_vector_load_dir = load_dirs[i] if i < len(load_dirs) else load_dirs[0]
        hparams_dict[alg_name] = hparams

    return hparams_dict
