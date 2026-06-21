"""
SAE-feature vector generator subpackage.

Kept lightweight for config-only workflows (e.g. hyperparameter loading).

We try to re-export `search_for_explanations` / `get_feature_description`
for backward compatibility (some tutorials import them at the package level),
but guard the import so that missing optional dependencies (e.g. sae_lens,
dotenv) never break lightweight imports such as `import steer.utils`.
"""

try:  # optional, heavy dependencies (sae_lens / dotenv)
    from .generate_sae_feature_vectors import (
        search_for_explanations,
        get_feature_description,
    )
    __all__ = ["search_for_explanations", "get_feature_description"]
except Exception:
    __all__ = []
