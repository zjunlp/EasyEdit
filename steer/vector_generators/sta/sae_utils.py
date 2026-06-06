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

sys.path.append("../")
import pdb

def load_sae_from_dir(sae_dir: Path | str, device: str = "cpu") -> SAE:
    """
    Load an SAE saved to a directory (``cfg.json`` + ``sae_weights.safetensors``) together
    with its log-sparsity tensor.
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

    return sae, log_sparsity


def clear_gpu_cache(cache):
    del cache
    torch.cuda.empty_cache()

def load_gemma_2_sae(
    sae_path: str,
    device: str = "cpu",
    repo_id: str = "gemma-scope-9b-it-res",
    force_download: bool = False,
    cfg_overrides: Optional[Dict[str, Any]] = None,
    d_sae_override: Optional[int] = None,
    layer_override: Optional[int] = None,
) -> Tuple[Dict[str, Any], Dict[str, torch.Tensor], Optional[torch.Tensor]]:
    """
    Custom loader for Gemma 2 SAEs. Not sure if this should be preserved.
    """
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

    cfg_dict = handle_config_defaulting(cfg_dict)
    sae = SAE.from_dict(cfg_dict)
    sae.load_state_dict(state_dict)

    # No sparsity tensor for Gemma 2 SAEs
    log_sparsity = None

    return sae, log_sparsity


if __name__ == "__main__":

    sae_path = "/mnt/20t/msy/models/gemma-scope-9b-it-res/layer_20/width_131k/average_l0_24"
    sae, _ = load_gemma_2_sae(sae_path=sae_path, device="cuda:5")
    llama_sae, _ = load_sae_from_dir("/mnt/20t/msy/shae/exp/llama-3.1-jumprelu-resid_post/layer_20/ef16")
    import pdb;pdb.set_trace()
