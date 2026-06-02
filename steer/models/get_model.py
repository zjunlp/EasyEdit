from .model_wrapper import GPTWrapper, GemmaWrapper, LlamaWrapper, QwenWrapper
import torch as t


def _load_llava_wrapper():
    # Lazy import: the multimodal wrapper pulls in heavier deps (alg_dict, AutoProcessor)
    # at module load time, so only import it when a multimodal model is actually
    # requested. This keeps the text-only path light and avoids import cycles.
    from .Multimodalmodel_wrapper import LlavaOVWrapper
    return LlavaOVWrapper


# A registry of model name and their corresponding wrapper resolvers. The resolver is 
# either a Wrapper class or a zero-arg callable returning one (used for multimodal 
# wrappers that are imported lazily that saves the importing time).
# Each entry is (keyword, wrapper_resolver, is_multimodal):
#   - keyword        : matched as a substring of the lowercased model_name_or_path.
#   - wrapper_resolver: a Wrapper class, or a zero-arg callable returning one.
#   - is_multimodal  : whether the model needs an image processor.
# match order is match priority. So put more specific / multimodal keys first so a name like
# "qwen2.5-vl" is recognized as multimodal before the generic text "qwen".
MODEL_REGISTRY = [
    ('llava', _load_llava_wrapper, True),
    ('llama', LlamaWrapper, False),
    ('gpt',   GPTWrapper,   False),
    ('gemma', GemmaWrapper, False),
    ('qwen',  QwenWrapper,  False),
    ('qwq',   QwenWrapper,  False),
]


def resolve_model_entry(model_name_or_path):
    """Resolve a model name to (wrapper_class, is_multimodal) without instantiating it.

    Raises ValueError if no registry entry matches.
    """
    name = model_name_or_path.lower()
    for keyword, resolver, is_multimodal in MODEL_REGISTRY:
        if keyword in name:
            wrapper_cls = resolver if isinstance(resolver, type) else resolver()
            return wrapper_cls, is_multimodal
    raise ValueError(f"model_name_or_path {model_name_or_path} not supported")


def get_model(hparams):

    torch_dtype = hparams.torch_dtype if hasattr(hparams, "torch_dtype") else t.float32
    use_cache = hparams.use_cache if hasattr(hparams, "use_cache") else True
    use_chat = hparams.use_chat_template if hasattr(hparams, "use_chat_template") else False
    device = hparams.device if hasattr(hparams, "device") else "cuda" if t.cuda.is_available() else "cpu"
    model_name_or_path = hparams.model_name_or_path if hasattr(hparams, "model_name_or_path") else None
    override_model_weights_path = hparams.override_model_weights_path if hasattr(hparams, "override_model_weights_path") else None

    wrapper_cls, is_multimodal = resolve_model_entry(hparams.model_name_or_path)

    kwargs = dict(
        torch_dtype=torch_dtype,
        use_chat=use_chat,
        device=device,
        model_name_or_path=model_name_or_path,
        override_model_weights_path=override_model_weights_path,
        hparams=hparams,
    )
    # The multimodal wrapper's constructor does not (yet) accept use_cache; only pass it to
    # text wrappers. Step 2 (wrapper dedup) harmonizes the signatures so this branch goes away.
    if not is_multimodal:
        kwargs["use_cache"] = use_cache

    model = wrapper_cls(**kwargs)

    # Returns a 2-tuple for both modalities. The wrapper shows if the model is multimodal with attr
    # tokenizer/processor: text wrappers expose model.processor = None, multimodal wrappers
    # expose a model.processor. Downstream detects modality via model.processor.
    return model, model.tokenizer
