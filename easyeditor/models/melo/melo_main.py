from typing import Any, Dict, List, Tuple
import torch
from copy import deepcopy
from transformers import AutoModelForCausalLM, AutoTokenizer
from .melo_hparams import MELOHyperParams
from .util import get_tokenizer
from .melo import LORA
from ...util import nethook


def apply_melo_to_model(
        model: AutoModelForCausalLM,
        tok: AutoTokenizer,
        requests: List[Dict],
        hparams: MELOHyperParams,
        copy=False,
        return_orig_weights=False,
        keep_original_weight=False,
        **kwargs: Any,
) -> Tuple[AutoModelForCausalLM, Dict[str, Any]]:
    model = deepcopy(model)
    # only support single edit.we will support sequence edit soon
    weights_copy = {}
    device = torch.device(f'cuda:{hparams.device}')
    tokenizer = get_tokenizer(hparams)
    
    editor = LORA(model,hparams,tokenizer)
    tokens = tokenizer(requests[0],tok,device)
    editor.to(device)
    editor.edit(tokens)
    return editor.model,weights_copy


