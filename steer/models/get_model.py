from .model_wrapper import GPTWrapper,GemmaWrapper,LlamaWrapper,QwenWrapper
import torch as t
from vllm import LLM

def get_model(hparams):

    torch_dtype = hparams.torch_dtype if hasattr(hparams, "torch_dtype") else t.float32
    use_cache = hparams.use_cache if hasattr(hparams, "use_cache") else True
    use_chat = hparams.use_chat_template if hasattr(hparams, "use_chat_template") else False
    device = hparams.device if hasattr(hparams, "device") else "cuda" if t.cuda.is_available() else "cpu"
    model_name_or_path = hparams.model_name_or_path if hasattr(hparams, "model_name_or_path") else None
    override_model_weights_path = hparams.override_model_weights_path if hasattr(hparams, "override_model_weights_path") else None
    vllm_enable = hparams.vllm_enable if hasattr(hparams, "vllm_enable") else False
    
    if vllm_enable:
        VLLM_model = LLM(
                model=model_name_or_path,
                #enable_steer_vector=True,
                enforce_eager=True,
                tensor_parallel_size=1,
                dtype=torch_dtype,
        )
    else:
        VLLM_model = None

    if 'llama' in hparams.model_name_or_path.lower():

        model = LlamaWrapper( 
            torch_dtype=torch_dtype,
            use_chat=use_chat,
            device=device,
            model_name_or_path=model_name_or_path,
            use_cache=use_cache,
            override_model_weights_path=override_model_weights_path,
            hparams=hparams,
            VLLM_model=VLLM_model
            )
        if vllm_enable:
            return model, VLLM_model
        else:
            return model, model.tokenizer
    
    elif 'gpt' in hparams.model_name_or_path.lower():

        model = GPTWrapper( 
            torch_dtype=torch_dtype,
            use_chat=use_chat,
            device=device,
            model_name_or_path=model_name_or_path,
            use_cache=use_cache,
            override_model_weights_path=override_model_weights_path,
            hparams=hparams,
            VLLM_model=VLLM_model
            )
        if vllm_enable:
            return model, VLLM_model
        else:
            return model, model.tokenizer
    
    elif 'gemma' in hparams.model_name_or_path.lower():
    
        model = GemmaWrapper(  
            torch_dtype=torch_dtype,
            use_chat=use_chat,
            device=device,
            model_name_or_path=model_name_or_path,
            use_cache=use_cache,
            override_model_weights_path=override_model_weights_path,
            hparams=hparams,
            VLLM_model=VLLM_model
            )
        if vllm_enable:
            return model, VLLM_model
        else:
            return model, model.tokenizer
    
    elif 'qwen' in hparams.model_name_or_path.lower() or 'qwq' in hparams.model_name_or_path.lower():

        model = QwenWrapper(  
            torch_dtype=torch_dtype,
            use_chat=use_chat,
            device=device,
            model_name_or_path=model_name_or_path,
            use_cache=use_cache,
            override_model_weights_path=override_model_weights_path,
            hparams=hparams,
            VLLM_model=VLLM_model
            )
        if vllm_enable:
            return model, VLLM_model
        else:
            return model, model.tokenizer
    
    else:
        raise ValueError(f"model_name_or_path {hparams.model_name_or_path} not supported")
    

