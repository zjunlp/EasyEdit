from ..vector_generators import(
    SaeFeatureHyperParams,
    STAHyperParams,
    VectorPromptHyperParams,
    CAAHyperParams,
    LmSteerHyperParams,
    MergeVectorHyperParams
)
from ..vector_appliers import(
    ApplySaeFeatureHyperParams,
    ApplySTAHyperParams,
    ApplyCAAHyperParams,
    ApplyVectorPromptHyperParams,
    ApplyLmSteerHyperParams,
    ApplyPromptHyperParams,
    ApplyMergeVectorHyperParams,
)

from ..vector_generators import (
    generate_lm_steer_delta,
    generate_caa_vectors,
    generate_vector_prompt_vectors,
    generate_sae_feature_vectors,
    generate_sta_vectors,
    generate_merge_vector,
)
from ..vector_appliers import (
     apply_lm_steer,
     apply_caa,
     apply_vector_prompt,
     apply_sae_feature,
     apply_sta,
     apply_prompt,
     apply_merge_vector,
)
import torch
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
    'merge_vector': {'train': MergeVectorHyperParams, 'apply': ApplyMergeVectorHyperParams}
}

METHODS_CLASS_DICT = {
    'lm_steer': {'train': generate_lm_steer_delta, 'apply': apply_lm_steer},
    'caa': {'train': generate_caa_vectors, 'apply': apply_caa},
    'vector_prompt': {'train': generate_vector_prompt_vectors, 'apply': apply_vector_prompt},
    'sae_feature': {'train': generate_sae_feature_vectors, 'apply': apply_sae_feature},
    'sta': {'train': generate_sta_vectors, 'apply': apply_sta},
    'prompt': {'apply': apply_prompt},
    'merge_vector': {'train': generate_merge_vector, 'apply': apply_merge_vector}
}