"""
Central registry for method hyperparams and implementations.

Important: keep this module lightweight to import.
We avoid importing full generator/applier implementations at import time because
they may pull in optional dependencies (datasets/dotenv/...) which are not
required for config parsing.
"""

from __future__ import annotations

import importlib
import torch

# HyperParam dataclasses (safe, lightweight imports)
from ..vector_generators.caa.generate_caa_hparam import CAAHyperParams
from ..vector_generators.lm_steer.generate_lm_steer_hparam import LmSteerHyperParams
from ..vector_generators.merge.generate_merge_vector_hparams import MergeVectorHyperParams
from ..vector_generators.reps.generate_reps_hparams import RePSHyperParams
from ..vector_generators.sae_feature.generate_sae_feature_hparam import SaeFeatureHyperParams
from ..vector_generators.sft.generate_sft_hparams import SFTHyperParams
from ..vector_generators.spilt.generate_spilt_hparams import SpiltHyperParams
from ..vector_generators.sta.generate_sta_hparam import STAHyperParams
from ..vector_generators.vector_prompt.generate_vector_prompt_hparam import VectorPromptHyperParams

from ..vector_appliers.caa.apply_caa_hparam import ApplyCAAHyperParams
from ..vector_appliers.lm_steer.apply_lm_steer_hparam import ApplyLmSteerHyperParams
from ..vector_appliers.merge.apply_merge_vector_hparam import ApplyMergeVectorHyperParams
from ..vector_appliers.prompt.apply_prompt_hparam import ApplyPromptHyperParams
from ..vector_appliers.reps.apply_reps_hparam import ApplyRepsHyperParams
from ..vector_appliers.sae_feature.apply_sae_feature_hparam import ApplySaeFeatureHyperParams
from ..vector_appliers.sft.apply_sft_hparam import ApplySFTHyperParams
from ..vector_appliers.spilt.apply_spilt_hparam import ApplySpiltHyperParams
from ..vector_appliers.sta.apply_sta_hparam import ApplySTAHyperParams
from ..vector_appliers.vector_prompt.apply_vector_prompt_hparam import ApplyVectorPromptHyperParams

DTYPES_DICT ={
    "float16": torch.float16,
    "float32": torch.float32,
    "float64": torch.float64,
    "bfloat16": torch.bfloat16,
    'float64': torch.float64,
    "bf16": torch.bfloat16,
    'fp16': torch.float16,
    'fp32': torch.float32,
    'fp64': torch.float64
}
HYPERPARAMS_CLASS_DICT = {
    'lm_steer': {'train': LmSteerHyperParams, 'apply': ApplyLmSteerHyperParams},
    'sae_feature': {'train': SaeFeatureHyperParams, 'apply': ApplySaeFeatureHyperParams},
    'sta': {'train': STAHyperParams, 'apply': ApplySTAHyperParams},
    'vector_prompt': {'train': VectorPromptHyperParams, 'apply': ApplyVectorPromptHyperParams},
    'caa': {'train': CAAHyperParams, 'apply': ApplyCAAHyperParams},
    'prompt': {'apply': ApplyPromptHyperParams},
    'merge_vector': {'train': MergeVectorHyperParams, 'apply': ApplyMergeVectorHyperParams},
    'reps':{'train': RePSHyperParams, 'apply': ApplyRepsHyperParams},
    'sft':{'train': SFTHyperParams, 'apply': ApplySFTHyperParams},
    'spilt':{'train': SpiltHyperParams, 'apply': ApplySpiltHyperParams}
}

# Lazy import targets: "module.path:function_name"
METHOD_FNS = {
    "lm_steer": {
        "train": "steer.vector_generators.lm_steer.generate_lm_steer_delta:generate_lm_steer_delta",
        "apply": "steer.vector_appliers.lm_steer.apply_lm_steer:apply_lm_steer",
    },
    "caa": {
        "train": "steer.vector_generators.caa.generate_caa_vectors:generate_caa_vectors",
        "apply": "steer.vector_appliers.caa.apply_caa:apply_caa",
    },
    "vector_prompt": {
        "train": "steer.vector_generators.vector_prompt.generate_vector_prompt_vectors:generate_vector_prompt_vectors",
        "apply": "steer.vector_appliers.vector_prompt.apply_vector_prompt:apply_vector_prompt",
    },
    "sae_feature": {
        "train": "steer.vector_generators.sae_feature.generate_sae_feature_vectors:generate_sae_feature_vectors",
        "apply": "steer.vector_appliers.sae_feature.apply_sae_feature:apply_sae_feature",
    },
    "sta": {
        "train": "steer.vector_generators.sta.generate_sta_vectors:generate_sta_vectors",
        "apply": "steer.vector_appliers.sta.apply_sta:apply_sta",
    },
    "prompt": {
        "apply": "steer.vector_appliers.prompt.apply_prompt:apply_prompt",
    },
    "merge_vector": {
        "train": "steer.vector_generators.merge.generate_merge_vector:generate_merge_vector",
        "apply": "steer.vector_appliers.merge.apply_merge_vector:apply_merge_vector",
    },
    "reps": {
        "train": "steer.vector_generators.reps.generate_reps:generate_reps",
        "apply": "steer.vector_appliers.reps.apply_reps:apply_reps",
    },
    "sft": {
        "train": "steer.vector_generators.sft.generate_sft:generate_sft",
        "apply": "steer.vector_appliers.sft.apply_sft:apply_sft",
    },
    "spilt": {
        "train": "steer.vector_generators.spilt.generate_spilt:generate_spilt",
        "apply": "steer.vector_appliers.spilt.apply_spilt:apply_spilt",
    },
}


def get_method_fn(alg_name: str, phase: str):
    """phase is 'train' or 'apply'."""
    if alg_name not in METHOD_FNS or phase not in METHOD_FNS[alg_name]:
        raise NotImplementedError(f"Method {alg_name} phase {phase} not implemented!")
    target = METHOD_FNS[alg_name][phase]
    mod_path, fn_name = target.split(":", 1)
    mod = importlib.import_module(mod_path)
    return getattr(mod, fn_name)


from collections.abc import Mapping


class _LazyMethodPhases(Mapping):
    """
    Backward-compatible, dict-like view for a single method, e.g.::

        METHODS_CLASS_DICT["caa"]["apply"]  # -> apply_caa (imported lazily)

    The underlying train/apply function is imported only on access, so simply
    referencing ``METHODS_CLASS_DICT`` never pulls optional/heavy dependencies.
    """

    def __init__(self, alg_name: str):
        self._alg_name = alg_name

    def __getitem__(self, phase: str):
        if phase not in METHOD_FNS.get(self._alg_name, {}):
            raise KeyError(phase)
        return get_method_fn(self._alg_name, phase)

    def __iter__(self):
        return iter(METHOD_FNS.get(self._alg_name, {}))

    def __len__(self):
        return len(METHOD_FNS.get(self._alg_name, {}))


class _LazyMethodsDict(Mapping):
    """
    Backward-compatible replacement for the old eager ``METHODS_CLASS_DICT``.

    Usage stays identical::

        METHODS_CLASS_DICT[alg_name]["train"](...)
        METHODS_CLASS_DICT[alg_name]["apply"](...)
        alg_name in METHODS_CLASS_DICT

    but the concrete implementations are resolved lazily via ``get_method_fn``.
    """

    def __getitem__(self, alg_name: str):
        if alg_name not in METHOD_FNS:
            raise KeyError(alg_name)
        return _LazyMethodPhases(alg_name)

    def __iter__(self):
        return iter(METHOD_FNS)

    def __len__(self):
        return len(METHOD_FNS)

    def __contains__(self, alg_name):
        return alg_name in METHOD_FNS


# Backward-compatible lazy registry (kept for existing callers such as demos).
METHODS_CLASS_DICT = _LazyMethodsDict()

VLLM_SUPPORTED_METHODS = ['caa']