from .model_wrapper import GPTWrapper, GemmaWrapper, LlamaWrapper, QwenWrapper
import torch as t
from jinja2 import Template
from pathlib import Path

# lazy import wrappers for VL families, only when needed.
def _load_llava_wrapper():
    from .Multimodalmodel_wrapper import LlavaOnevisionWrapper
    return LlavaOnevisionWrapper


def _load_qwen_vl_wrapper():
    from .Multimodalmodel_wrapper import QwenVLWrapper
    return QwenVLWrapper


def _load_gemma_vl_wrapper():
    from .Multimodalmodel_wrapper import GemmaVLWrapper
    return GemmaVLWrapper


# VL family key -> lazy wrapper resolver.
_VL_FAMILY_RESOLVERS = {
    "qwen_vl": _load_qwen_vl_wrapper,
    "gemma_vl": _load_gemma_vl_wrapper,
    "llava": _load_llava_wrapper,
}

# Text wrapper, order = priority for substring matching in the model name.
_TEXT_REGISTRY = [
    ("llama", LlamaWrapper),
    ("gpt", GPTWrapper),
    ("gemma", GemmaWrapper),
    ("qwen", QwenWrapper),
    ("qwq", QwenWrapper),
]


def _vl_family_from_model_type(model_type: str, archs: str):
    blob = f"{model_type} {archs}"
    if "qwen" in blob:
        return "qwen_vl"
    if "gemma" in blob or "paligemma" in blob:
        return "gemma_vl"
    if "llava" in blob:
        return "llava"
    # Unknown vision-language model: fall back to the most permissive (LLaVA-style) wrapper.
    return "llava"


def _detect_from_config(model_name_or_path):
    """Using auto config instead of the name to detect if the model is multimodal."""
    try:
        from transformers import AutoConfig
        cfg = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=True)
    except Exception:
        return None, False

    model_type = (getattr(cfg, "model_type", "") or "").lower()
    archs = " ".join(getattr(cfg, "architectures", None) or []).lower()
    has_vision = getattr(cfg, "vision_config", None) is not None

    is_multimodal = has_vision or "imagetexttotext" in archs
    if not is_multimodal:
        return None, True
    return _vl_family_from_model_type(model_type, archs), True


def _detect_vl_family_from_name(name: str):
    """Using name keywords to detect the VL family."""
    if "qwen" in name and "vl" in name:
        return "qwen_vl"
    if "paligemma" in name:
        return "gemma_vl"
    # Gemma VL checkpoints are gemma-3/3n/4 with a vision encoder; the text-only ones
    # (e.g. gemma-2-*, gemma-3-1b) are smaller and explicitly excluded.
    if "gemma" in name and ("vl" in name or "gemma-3" in name or "gemma3" in name
                            or "gemma-4" in name or "gemma4" in name) and "1b" not in name:
        return "gemma_vl"
    if "llava" in name or "-vl" in name or "_vl" in name:
        return "llava"
    return None


def resolve_model_entry(model_name_or_path):
    """Using a combination of config inspection and name substring matching to resolve the model wrapper class."""
    family, config_read = _detect_from_config(model_name_or_path)

    # If the config could not be read, try to spot a VL family from the name.
    if family is None and not config_read:
        family = _detect_vl_family_from_name(model_name_or_path.lower())

    if family is not None:
        return _VL_FAMILY_RESOLVERS[family](), True

    # Text models: substring match against the text registry.
    name = model_name_or_path.lower()
    for keyword, resolver in _TEXT_REGISTRY:
        if keyword in name:
            wrapper_cls = resolver if isinstance(resolver, type) else resolver()
            return wrapper_cls, False
    raise ValueError(f"model_name_or_path {model_name_or_path} not supported")

def load_chat_template(template_name: str) -> str:
    """
    for models that lacks of a built-in chat template, you can load a jinja template from the local chat_template folder and inject it into the tokenizer.
    Some experiments results may fail to reproduce in current dependecies, this is due to the upgrade of transformers library, which remove the built-in chat template for some models.
    You can load the chat template from local and inject it into the tokenizer to make the model response in a chat way, and then you can find the experiment result can be reproduced.
    """
    current_file = Path(__file__).resolve()
    
    template_path = current_file.parent / "chat_template" / template_name
    
    if not template_path.exists():
        raise FileNotFoundError(f"Unable to find template file: {template_path}")
        
    with open(template_path, "r", encoding="utf-8") as f:
        return f.read()


def get_model(hparams):
    """Resolves and instantiates a model wrapper based on the provided hyperparameters."""
    # Update for torch_dtype since torch_dtype will be deprecated in torch in the future.
    if hasattr(hparams, "dtype") and hparams.dtype is not None:
        dtype = hparams.dtype
    elif hasattr(hparams, "torch_dtype") and hparams.torch_dtype is not None:
        dtype = hparams.torch_dtype
    else:
        dtype = t.float32
    use_cache = hparams.use_cache if hasattr(hparams, "use_cache") else True
    use_chat = hparams.use_chat_template if hasattr(hparams, "use_chat_template") else False
    device = hparams.device if hasattr(hparams, "device") else "cuda" if t.cuda.is_available() else "cpu"
    model_name_or_path = hparams.model_name_or_path if hasattr(hparams, "model_name_or_path") else None
    override_model_weights_path = hparams.override_model_weights_path if hasattr(hparams, "override_model_weights_path") else None

    wrapper_cls, _ = resolve_model_entry(hparams.model_name_or_path)

    # Text and multimodal wrappers share a constructor signature, so the same kwargs work
    # for both.
    model = wrapper_cls(
        dtype=dtype,
        use_chat=use_chat,
        device=device,
        model_name_or_path=model_name_or_path,
        use_cache=use_cache,
        override_model_weights_path=override_model_weights_path,
        hparams=hparams,
    )
    # Returns a 2-tuple for both modalities. Downstream detects modality via model.processor:
    # text wrappers expose model.processor = None, multimodal wrappers expose a real processor.
    if model.tokenizer.pad_token is None:
        model.tokenizer.pad_token = model.tokenizer.eos_token
        model.tokenizer.pad_token_id = model.tokenizer.eos_token_id
    return model, model.tokenizer
