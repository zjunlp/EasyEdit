from .model_wrapper import GPTWrapper, GemmaWrapper, LlamaWrapper, QwenWrapper
import torch as t
from jinja2 import Template
from pathlib import Path

# ---- lazy resolvers for the multimodal family wrappers ---------------------------------
# These pull heavier deps (AutoProcessor, PIL) at import time, so import them only when a
# multimodal model is actually requested. Each returns the wrapper CLASS.
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

# Text wrapper fallback, matched as a substring of the lowercased model path. Order = priority.
_TEXT_REGISTRY = [
    ("llama", LlamaWrapper),
    ("gpt", GPTWrapper),
    ("gemma", GemmaWrapper),
    ("qwen", QwenWrapper),
    ("qwq", QwenWrapper),
]


def _vl_family_from_model_type(model_type: str, archs: str):
    """Map a HF ``model_type`` / architectures string to one of our VL family keys."""
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
    """Inspect the HF config (no weights) -> (vl_family | None, config_was_read).

    Architecture-accurate: a model is multimodal iff its config carries a ``vision_config``
    (or an image-text-to-text architecture). This distinguishes e.g. gemma-3-1b (text, no
    vision_config) from gemma-3-4b (VL) — something name matching cannot do.

    Returns ``(family, True)`` for a recognized VL model, ``(None, True)`` for a text model
    whose config we COULD read, and ``(None, False)`` if the config could not be read (so the
    caller knows to fall back to name keywords).
    """
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
    """Name-keyword fallback used only when the config is unavailable. Returns a VL family
    key or ``None`` (let the text registry handle it)."""
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
    """Resolve a model name to ``(wrapper_class, is_multimodal)`` without instantiating it.

    Strategy: config-first (architecture-accurate), then name-keyword fallback. Multimodal
    models route to the matching VL family wrapper; everything else to a text wrapper.
    Raises ``ValueError`` if nothing matches.
    """
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
    for models that lacks of a built-in chat template, we load a jinja template from the local chat_template folder and inject it into the tokenizer.
    """
    current_file = Path(__file__).resolve()
    
    template_path = current_file.parent / "chat_template" / template_name
    
    if not template_path.exists():
        raise FileNotFoundError(f"Unable to find template file: {template_path}")
        
    with open(template_path, "r", encoding="utf-8") as f:
        return f.read()


def get_model(hparams):

    torch_dtype = hparams.torch_dtype if hasattr(hparams, "torch_dtype") else t.float32
    use_cache = hparams.use_cache if hasattr(hparams, "use_cache") else True
    use_chat = hparams.use_chat_template if hasattr(hparams, "use_chat_template") else False
    device = hparams.device if hasattr(hparams, "device") else "cuda" if t.cuda.is_available() else "cpu"
    model_name_or_path = hparams.model_name_or_path if hasattr(hparams, "model_name_or_path") else None
    override_model_weights_path = hparams.override_model_weights_path if hasattr(hparams, "override_model_weights_path") else None

    wrapper_cls, _ = resolve_model_entry(hparams.model_name_or_path)

    # Text and multimodal wrappers share a constructor signature, so the same kwargs work
    # for both.
    model = wrapper_cls(
        torch_dtype=torch_dtype,
        use_chat=use_chat,
        device=device,
        model_name_or_path=model_name_or_path,
        use_cache=use_cache,
        override_model_weights_path=override_model_weights_path,
        hparams=hparams,
    )
    if model.tokenizer.chat_template is None:
        if model.processor is not None:
            if "qwen" in model_name_or_path.lower():
                template_name = "qwen_vl.jinja"
            elif "gemma" in model_name_or_path.lower():
                template_name = "gemma_vl.jinja"
            else:
                raise ValueError(f"No chat template found for model {model_name_or_path}")
        else:
            template_name = "default.jinja"

        try:
            jinja_template = load_chat_template(template_name)
            model.tokenizer.chat_template = jinja_template
        except Exception as e:
            print(f"Failed to load template, keeping default settings, may cause unexpected behavior due to lack of default chat template after Transformers upgraded to Version 5.x. Error: {e}")

    # Returns a 2-tuple for both modalities. Downstream detects modality via model.processor:
    # text wrappers expose model.processor = None, multimodal wrappers expose a real processor.
    if model.tokenizer.pad_token is None:
        model.tokenizer.pad_token = model.tokenizer.eos_token
        model.tokenizer.pad_token_id = model.tokenizer.eos_token_id
    return model, model.tokenizer
