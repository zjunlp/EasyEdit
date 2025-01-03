from typing import Any, Dict, List, Tuple
import torch
from copy import deepcopy
from transformers import AutoModelForCausalLM, AutoTokenizer
from .DEFER import DEFER
from .defer_hparams import DeferHyperParams
from .utils import tokenize
from ...util import nethook


def apply_defer_to_model(
        model: AutoModelForCausalLM,
        tok: AutoTokenizer,
        requests: List[Dict],
        hparams: DeferHyperParams,
        copy=False,
        return_orig_weights=False,
        keep_original_weight=False,
        **kwargs: Any,
) -> Tuple[AutoModelForCausalLM, Dict[str, Any]]:
    request = requests[0]
    if copy:
        model = deepcopy(model)
    device = torch.device(f'cuda:{hparams.device}')
    # print(model)
    if not isinstance(model, DEFER):
        editor = DEFER(model=model, config=hparams)
    else:
        editor = model
    tokens = tokenize(request, tokenizer=tok, device=device)
    editor.edit(config=hparams, tokens=tokens,)# edit_id=request['target_new'])
            
    weights_copy = editor.reset_layer


    return editor, weights_copy


