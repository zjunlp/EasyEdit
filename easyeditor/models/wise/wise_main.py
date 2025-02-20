from typing import Any, Dict, List, Tuple
from copy import deepcopy
from transformers import AutoModelForCausalLM, AutoTokenizer
from .WISE import WISE, WISEMultimodal
from .utils import tokenize, multimodal_tokenize, get_context_templates
from .wise_hparams import WISEHyperParams
WISEload = True
def apply_wise_to_model(
        model: AutoModelForCausalLM,
        tok: AutoTokenizer,
        requests: List[Dict],
        hparams: WISEHyperParams,
        copy=False,
        **kwargs: Any,
) -> Tuple[AutoModelForCausalLM, Dict[str, Any]]:
    if copy:
        model = deepcopy(model)
    device = f'cuda:{hparams.device}'
    context_templates = get_context_templates(model, tok, length_params=[[5,5], [10,5]], device=device)
    editor = WISE(model=model, config=hparams, device=device)
    import os
    global WISEload
    if hasattr(hparams, 'load_path') and hparams.load_path and os.path.exists(hparams.load_path) and WISEload:
        print("Start loading the WISE model!")
        editor.load(hparams.load_path)
        WISEload=False
    print(f"Executing WISE algorithm for the update: ")
    for request in requests:
        print(
            f"[{request['prompt']}] -> [{request['target_new']}]"
        )
    tokens, act_mask, deact_mask = tokenize(requests, tokenizer=tok, device=device, context_templates=context_templates, hparams=hparams)
    editor.edit(config=hparams, tokens=tokens, act_mask=act_mask, deact_mask=deact_mask)

    weights_copy = editor.reset_layer

    return editor, weights_copy


def apply_wise_to_multimodal_model(
        model: AutoModelForCausalLM,
        tok: AutoTokenizer,
        requests: List[Dict],
        hparams: WISEHyperParams,
        copy=False,
        **kwargs: Any,
) -> Tuple[AutoModelForCausalLM, Dict[str, Any]]:
    device = f'cuda:{hparams.device}'    
    if copy:
        model = deepcopy(model)
        model.to(device)

    editor = WISEMultimodal(model=model, config=hparams, device=device)
    import os
    global WISEload
    if hasattr(hparams, 'load_path') and hparams.load_path and os.path.exists(hparams.load_path) and WISEload:
        print("Start loading the WISE model!")
        editor.load(hparams.load_path)
        WISEload=False
    print(f"Executing WISE algorithm for the update: ")
    for request in requests:
        print(
            f"[{request['prompt']}] -> [{request['target']}]"
        )

    multimodal_inputs, text_tokens, ans_token_len, act_mask, deact_mask = multimodal_tokenize(requests, processor=tok, device=device, context_templates=None, hparams=hparams)
    editor.edit(config=hparams, multimodal_inputs=multimodal_inputs, ans_token_len=ans_token_len, text_tokens=text_tokens, act_mask=act_mask, deact_mask=deact_mask)
    weights_copy = editor.reset_layer
    return editor, weights_copy


