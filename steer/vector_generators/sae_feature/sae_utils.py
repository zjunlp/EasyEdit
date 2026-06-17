from typing import Any, Dict, Optional, Tuple
import os, sys
import torch
from pathlib import Path
import numpy as np
from sae_lens import SAE
from sae_lens.loading.pretrained_sae_loaders import (
    get_gemma_2_config_from_hf,
    handle_config_defaulting,
)
from safetensors import safe_open
"""To support the new sae_lens, we almost rewrite the whole loading logic for SAEs,
and for different formats of SAEs (sae-lens directory, Gemma-Scope npz, Qwen-Scope per-layer pt), we have separate loading functions.
The main entry point is `load_sae`, which dispatches by on-disk format, not by model name, so it can be used for any model as long as the SAE is in a recognised format.
"""
sys.path.append("../")


def _sae_cfg_dict(sae) -> Dict[str, Any]:
    """Return a plain, mutable dict for an SAE's config metadata.

    Downstream code both reads (``sae_cfg.get("neuronpedia_id")``) and writes
    (``sae_cfg["neuronpedia_id"] = ...``) this, so we always hand back a real dict regardless
    of whether ``sae.cfg.metadata`` is a Mapping, a dataclass, or absent.
    """
    md = getattr(getattr(sae, "cfg", None), "metadata", None)
    if md is None:
        return {}
    try:
        return dict(md)
    except (TypeError, ValueError):
        try:
            return dict(vars(md))
        except TypeError:
            return {}


def load_sae_from_dir(sae_dir: Path | str, device: str = "cpu") -> Tuple[SAE, Dict[str, Any], Optional[torch.Tensor]]:
    """
    Load an SAE saved to a directory (``cfg.json`` + ``sae_weights.safetensors``) together
    with its log-sparsity tensor. Returns ``(sae, sae_cfg, log_sparsity)``.
    """
    sae_dir = Path(sae_dir)
    # print(f"Loading SAE from {sae_dir}")

    if not all([x.is_file() for x in sae_dir.iterdir()]):
        raise ValueError(
            "Not all files are present in the directory! Only files allowed for loading SAE Directory."
        )

    sae = SAE.load_from_pretrained(sae_dir, device=device)
    assert sae is not None and isinstance(sae, SAE)

    with safe_open(os.path.join(sae_dir, "sparsity.safetensors"), framework="pt", device=device) as f:  # type: ignore
        log_sparsity = f.get_tensor("sparsity")

    return sae, _sae_cfg_dict(sae), log_sparsity


def clear_gpu_cache(cache):
    del cache
    torch.cuda.empty_cache()


def load_npz_sae(
    sae_path: str,
    device: str = "cpu",
    repo_id: str = "gemma-scope-9b-it-res",
    force_download: bool = False,
    cfg_overrides: Optional[Dict[str, Any]] = None,
    d_sae_override: Optional[int] = None,
    layer_override: Optional[int] = None,
) -> Tuple[SAE, Dict[str, Any], Optional[torch.Tensor]]:
    """Loader for a gemma-like SAE in the original npz format (``params.npz`` + HF repo config). Returns ``(sae, sae_cfg, log_sparsity)``.
    """
    # NOTE: get_gemma_2_config_from_hf is the sae_lens (>=6.x) name; confirm its signature on a
    # box with sae_lens installed if the release layout changes.
    cfg_dict = get_gemma_2_config_from_hf(repo_id, sae_path, d_sae_override, layer_override)
    cfg_dict["device"] = device
    if d_sae_override is not None:
        cfg_dict["d_sae"] = d_sae_override

    # Apply overrides if provided
    if cfg_overrides is not None:
        cfg_dict.update(cfg_overrides)

    # Load and convert the weights
    state_dict = {}
    with np.load(os.path.join(sae_path, "params.npz")) as data:
        for key in data.keys():
            state_dict_key = "W_" + key[2:] if key.startswith("w_") else key
            state_dict[state_dict_key] = (
                torch.tensor(data[key]).to(dtype=torch.float32).to(device)
            )

    # Handle scaling factor
    if "scaling_factor" in state_dict:
        if torch.allclose(
            state_dict["scaling_factor"], torch.ones_like(state_dict["scaling_factor"])
        ):
            del state_dict["scaling_factor"]
            cfg_dict["finetuning_scaling_factor"] = False
        else:
            assert cfg_dict[
                "finetuning_scaling_factor"
            ], "Scaling factor is present but finetuning_scaling_factor is False."
            state_dict["finetuning_scaling_factor"] = state_dict.pop("scaling_factor")
    else:
        cfg_dict["finetuning_scaling_factor"] = False

    # Upgrade the flat config dict to the 6.x schema then build the architecture-specific
    # SAE (jumprelu for Gemma Scope). Replaces the removed SAEConfig.from_dict + SAE(cfg).
    cfg_dict = handle_config_defaulting(cfg_dict)
    sae = SAE.from_dict(cfg_dict)
    sae.load_state_dict(state_dict)

    # No sparsity tensor for Gemma 2 SAEs
    log_sparsity = None

    return sae, cfg_dict, log_sparsity


class QwenScopeTopKSAE:
    """Loader for a Qwen-like SAE in the per-layer pt format (``layer{n}.sae.pt`` + optional ``config.json``). Returns ``(sae, sae_cfg, None)``.
    """

    def __init__(self, state: Dict[str, torch.Tensor], cfg: Dict[str, Any], device: str = "cpu"):
        self.cfg_dict = dict(cfg)
        self.device = device
        self.k = int(cfg.get("k", 100))
        # Keep everything in float32 (the released checkpoints are float32) for numerically stable
        # encode/decode regardless of the model's runtime dtype.
        self.W_enc = state["W_enc"].to(device=device, dtype=torch.float32)   # (d_sae, d_model)
        self.b_enc = state["b_enc"].to(device=device, dtype=torch.float32)   # (d_sae,)
        self.b_dec = state["b_dec"].to(device=device, dtype=torch.float32)   # (d_model,)
        self.W_dec = state["W_dec"].t().contiguous().to(device=device, dtype=torch.float32)  # (d_sae, d_model)
        self.d_sae, self.d_model = self.W_dec.shape

    def to(self, device):
        self.device = device
        for name in ("W_enc", "b_enc", "b_dec", "W_dec"):
            setattr(self, name, getattr(self, name).to(device))
        return self

    @torch.no_grad()
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Residual activations ``(..., d_model)`` -> sparse TopK feature acts ``(..., d_sae)``."""
        x = x.to(device=self.W_enc.device, dtype=self.W_enc.dtype)
        pre_acts = x @ self.W_enc.t() + self.b_enc
        vals, idx = pre_acts.topk(self.k, dim=-1)
        acts = torch.zeros_like(pre_acts)
        acts.scatter_(-1, idx, vals)
        return acts

    @torch.no_grad()
    def decode(self, acts: torch.Tensor) -> torch.Tensor:
        acts = acts.to(device=self.W_dec.device, dtype=self.W_dec.dtype)
        return acts @ self.W_dec + self.b_dec


def load_qwen_scope_sae(
    sae_path: str | Path,
    device: str = "cpu",
) -> Tuple[QwenScopeTopKSAE, Dict[str, Any], None]:
    """Load a single Qwen-Scope ``layer{n}.sae.pt`` checkpoint into a :class:`QwenScopeTopKSAE`.

    ``sae_path`` points at the per-layer ``.sae.pt`` file; the sibling ``config.json`` (if present)
    supplies ``k``/``d_sae``/``d_model``. Returns ``(sae, sae_cfg, log_sparsity=None)`` to match the
    signature of the other loaders.
    """
    import json

    sae_file = Path(sae_path)
    if not sae_file.is_file():
        raise FileNotFoundError(f"Qwen-Scope SAE file not found: {sae_file}")

    cfg: Dict[str, Any] = {}
    cfg_path = sae_file.parent / "config.json"
    if cfg_path.exists():
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)

    state = torch.load(sae_file, map_location=device)
    required = {"W_enc", "W_dec", "b_enc", "b_dec"}
    if not required.issubset(state):
        raise ValueError(
            f"Qwen-Scope SAE at '{sae_file}' is missing keys {required - set(state)}; "
            f"got {sorted(state)}."
        )

    sae = QwenScopeTopKSAE(state, cfg, device=device)
    sae_cfg: Dict[str, Any] = {
        "model_type": cfg.get("model_type", "topk_sae"),
        "d_sae": sae.d_sae,
        "d_model": sae.d_model,
        "k": sae.k,
        "hook_point": cfg.get("hook_point"),
        "base_model": cfg.get("base_model"),
    }
    return sae, sae_cfg, None


def load_sae(
    sae_path: str | Path,
    device: str = "cpu",
    release: Optional[str] = None,
) -> Tuple[SAE, Dict[str, Any], Optional[torch.Tensor]]:
    """Format-dispatching local SAE loader (model-agnostic). Returns ``(sae, sae_cfg, log_sparsity)``.

    Dispatch is by on-disk FORMAT, not by model name -- this answers "do other models need
    their own loader?": no. New families are handled either by the standard sae-lens directory
    layout below, or (for HF-hosted SAEs) by ``SAE.from_pretrained`` in the callers.
      - ``<path>/layer{n}.sae.pt`` -> Qwen-Scope TopK format (load_qwen_scope_sae)
      - ``<path>/params.npz``      -> Gemma-Scope npz format (load_npz_sae)
      - ``<path>/cfg.json``        -> standard sae-lens directory (load_sae_from_dir)
    """
    sae_dir = Path(sae_path)
    # Qwen-Scope: callers pass the per-layer file path directly (so the layer number appears in the
    # path, satisfying STA's `str(layer) in sae_paths[i]` assertion).
    if str(sae_path).endswith(".sae.pt") or (sae_dir.is_file() and sae_dir.suffix == ".pt"):
        return load_qwen_scope_sae(sae_path, device=device)
    if (sae_dir / "params.npz").exists():
        repo_id = release or "gemma-scope-9b-it-res"
        return load_npz_sae(str(sae_path), device=device, repo_id=repo_id)
    if (sae_dir / "cfg.json").exists():
        return load_sae_from_dir(sae_dir, device=device)
    raise ValueError(
        f"Unrecognized SAE format at '{sae_path}': expected layer*.sae.pt (Qwen-Scope), "
        f"params.npz (Gemma-Scope) or cfg.json (sae-lens directory)."
    )


if __name__ == "__main__":

    sae_path = "/mnt/20t/msy/models/gemma-scope-9b-it-res/layer_20/width_131k/average_l0_24"
    sae, _, _ = load_npz_sae(sae_path=sae_path, device="cuda:5")
    llama_sae, _, _ = load_sae_from_dir("/mnt/20t/msy/shae/exp/llama-3.1-jumprelu-resid_post/layer_20/ef16")
    import pdb;pdb.set_trace()
