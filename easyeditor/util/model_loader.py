"""Loaders for models that EasyEdit's string-matching CausalLM branch cannot handle.

Qwen3.8-27B is a VL-wrapped language model (`Qwen3_5ForConditionalGeneration`)
with a 64-layer hybrid-attention text backbone. The checkpoint path does not
contain ``vl``, so it must be detected from ``config.json`` rather than
``vl_utils`` aliases. Pure-text Qwen3.5-9B stays on the existing CausalLM path.
"""

from transformers import AutoConfig, AutoModelForImageTextToText, AutoTokenizer
import torch

from .device import normalize_device

QWEN35_VL_TEXT_ARCH = "Qwen3_5ForConditionalGeneration"


def _has_qwen38_path_hint(model_name):
    name = str(model_name or "").lower().replace("_", "-")
    return "qwen3.8" in name or "qwen3-8" in name


def _config_is_qwen35_vl_text(config):
    architectures = list(getattr(config, "architectures", None) or [])
    if QWEN35_VL_TEXT_ARCH in architectures:
        return True
    model_type = getattr(config, "model_type", None)
    if model_type != "qwen3_5":
        return False
    vision_config = getattr(config, "vision_config", None)
    text_config = getattr(config, "text_config", None)
    # Nested VL wrapper (vision + text). Pure-text Qwen3.5-9B has neither
    # Qwen3_5ForConditionalGeneration nor vision_config, so it stays False.
    return vision_config is not None and text_config is not None


def is_qwen35_vl_text_model(model_name):
    """Return True for Qwen3.5 VL-text checkpoints such as Qwen3.8-27B.

    Primary signal is ``AutoConfig`` (architectures / nested vision+text
    configs). A ``qwen3.8`` path substring is only used when config cannot be
    read, and is never the sole criterion so Qwen3.5-9B is not misclassified.
    """
    try:
        config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    except Exception:
        return _has_qwen38_path_hint(model_name)
    return _config_is_qwen35_vl_text(config)


def _fill_hidden_act(model):
    if getattr(model.config, "hidden_act", None):
        return
    text_config = getattr(model.config, "text_config", None)
    model.config.hidden_act = getattr(text_config, "hidden_act", "silu")


def _device_map_for(device):
    if device is None:
        return None
    dev = normalize_device(device)
    if dev.type == "cuda":
        index = 0 if dev.index is None else dev.index
        return {"": index}
    return {"": str(dev)}


def load_qwen35_language_model(
    model_name,
    device,
    torch_dtype=None,
    attn_implementation="eager",
):
    """Load the language backbone of a Qwen3.5 VL-text model.

    Forces bfloat16 (fp32 27B OOMs). Does not set tokenizer ``padding_side``;
    the editor still chooses padding per algorithm.
    """
    del torch_dtype  # this family is always loaded in bfloat16
    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    config.language_model_only = True
    model = AutoModelForImageTextToText.from_pretrained(
        model_name,
        config=config,
        torch_dtype=torch.bfloat16,
        device_map=_device_map_for(device),
        trust_remote_code=True,
        attn_implementation=attn_implementation,
    )
    model.eval()
    _fill_hidden_act(model)

    tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return model, tok
