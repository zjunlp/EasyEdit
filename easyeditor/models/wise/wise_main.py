from typing import Any, Dict, List, Tuple
from copy import deepcopy
from transformers import AutoModelForCausalLM, AutoTokenizer
from .WISE import WISE
from .utils import tokenize, get_context_templates
from .wise_hparams import WISEHyperParams

def apply_wise_to_model(
        model: AutoModelForCausalLM,
        tok: AutoTokenizer,
        requests: List[Dict],
        hparams: WISEHyperParams,
        copy=False,
        **kwargs: Any,
) -> Tuple[AutoModelForCausalLM, Dict[str, Any]]:
    request = requests[0]
    if copy:
        model = deepcopy(model)
    context_templates = get_context_templates(model, tok, length_params=[[5,5], [10,5]], device=hparams.device)
    editor = WISE(model=model, config=hparams, device=hparams.device)
    print(
        f"Executing WISE algorithm for the update: "
        f"[{request['prompt']}] -> [{request['target_new']}]"
    )
    tokens, act_mask, deact_mask = tokenize(request, tokenizer=tok, device=hparams.device, context_templates=context_templates, hparams=hparams)
    editor.edit(config=hparams, tokens=tokens, act_mask=act_mask, deact_mask=deact_mask)

    weights_copy = editor.reset_layer

    return editor, weights_copy


