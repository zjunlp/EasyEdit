import tempfile
from typing import Any, Dict, Optional, Tuple
import os, sys
import torch
from sae_lens import SAE
from pathlib import Path
import json
import shutil
import numpy as np
from sae_lens.toolkit.pretrained_sae_loaders import (
    gemma_2_sae_loader,
    get_gemma_2_config,
)
from sae_lens import SAE, SAEConfig, LanguageModelSAERunnerConfig, SAETrainingRunner
from safetensors import safe_open

sys.path.append("../")
import pdb

def load_sae_from_dir(sae_dir: Path | str, device: str = "cpu") -> SAE:
    """
    Due to a bug (https://github.com/jbloomAus/SAELens/issues/168) in the SAE save implementation for SAE Lens we need to make
    a specialized workaround.

    WARNING this will be creating a directory where the files are LINKED with the exception of "cfg.json" which is copied. This is NOT efficient
    and you should not be calling it many times!

    This wraps: https://github.com/jbloomAus/SAELens/blob/main/sae_lens/sae.py#L284.

    SPECIFICALLY fix cfg.json.
    """
    sae_dir = Path(sae_dir)
    # print(f"Loading SAE from {sae_dir}")

    if not all([x.is_file() for x in sae_dir.iterdir()]):
        raise ValueError(
            "Not all files are present in the directory! Only files allowed for loading SAE Directory."
        )

    # https://github.com/jbloomAus/SAELens/blob/9dacd4a9672c138b7c900ddd9a28d1b3b3a0870c/sae_lens/config.py#L188
    # Load ourselves instead of from_json because there are some __dir__ elements that are not in the JSON
    # They should ALL be enumerated in `derivatives`
    ##### BEGIN #####
    cfg_f = sae_dir / "cfg.json"
    with open(cfg_f, "r") as f:
        cfg = json.load(f)
    derivatives = [
        "tokens_per_buffer",
        "finetuning_scaling_factor",
    ]
    cfg["activation_fn"] = cfg.pop("activation_fn_str")
    derivative_values = []
    to_remove = []
    for x in derivatives:
        if x not in cfg:
            to_remove.append(x)
        else:
            derivative_values.append(cfg[x])
            del cfg[x]
    for x in to_remove:
        derivatives.remove(x)

    runner_config = LanguageModelSAERunnerConfig(**cfg)
    base_cfg = runner_config.get_base_sae_cfg_dict()

    for d, dv in zip(derivatives, derivative_values):
        assert d in base_cfg and base_cfg[d] == dv, f"Mismatch in {d}: expected {dv}, got {base_cfg.get(d)}"
    del derivative_values
    del derivatives
    del base_cfg
    ##### END #####

    # Load the SAE
    sae_config = runner_config.get_training_sae_cfg_dict()
    sae_config["d_sae"] = cfg["d_sae"]
    sae = None
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir = Path(temp_dir)

        # Copy in the CFG
        sae_config_f = temp_dir / "cfg.json"
        with open(sae_config_f, "w") as f:
            json.dump(sae_config, f)
        # Copy all the other files
        for name_f in sae_dir.iterdir():
            if name_f.name == "cfg.json":
                continue
            else:
                shutil.copy(name_f, temp_dir / name_f.name)
        # Load SAE
        sae = SAE.load_from_pretrained(temp_dir, device=device)
    assert sae is not None and isinstance(sae, SAE)

    with safe_open(os.path.join(sae_dir, "sparsity.safetensors"), framework="pt", device=device) as f:  # type: ignore
        log_sparsity = f.get_tensor("sparsity")

    return sae, sae_config, log_sparsity


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
    Custom loader for Gemma 2 SAEs.
    """
    cfg_dict = get_gemma_2_config(repo_id, sae_path, d_sae_override, layer_override)
    cfg_dict["device"] = device

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
    sae_cfg = SAEConfig.from_dict(cfg_dict)
    sae = SAE(sae_cfg)
    sae.load_state_dict(state_dict)

    # No sparsity tensor for Gemma 2 SAEs
    log_sparsity = None

    return sae, cfg_dict, log_sparsity


if __name__ == "__main__":

    sae_path = "/mnt/20t/msy/models/gemma-scope-9b-it-res/layer_20/width_131k/average_l0_24"
    sae, _ = load_gemma_2_sae(sae_path=sae_path, device="cuda:5")
    llama_sae, _ = load_sae_from_dir("/mnt/20t/msy/shae/exp/llama-3.1-jumprelu-resid_post/layer_20/ef16")
    import pdb;pdb.set_trace()
