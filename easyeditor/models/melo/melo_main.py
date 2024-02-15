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
    # only support single edit.we will support sequence edit soon
    if keep_original_weight:
        model=deepcopy(model)
    weights_copy = {}
    device = torch.device(f'cuda:{hparams.device}')
    tokenizer = get_tokenizer(hparams)
    if not isinstance(model,LORA):
        editor = LORA(model,hparams,tokenizer)
    else:
        editor = model
    tokens = tokenizer(requests[0],tok,device)
    editor.to(device)
    editor.edit(tokens)
    return editor,weights_copy
