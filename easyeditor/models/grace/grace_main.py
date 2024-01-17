from typing import Any, Dict, List, Tuple
import torch
from copy import deepcopy
from transformers import AutoModelForCausalLM, AutoTokenizer
from .GRACE import GRACE
from .grace_hparams import GraceHyperParams
from .utils import tokenize


def apply_grace_to_model(
        model: AutoModelForCausalLM,
        tok: AutoTokenizer,
        requests: List[Dict],
        hparams: GraceHyperParams,
        copy=False,
        return_orig_weights=False,
        keep_original_weight=False,
        **kwargs: Any,
) -> Tuple[AutoModelForCausalLM, Dict[str, Any]]:
    request = requests[0]
    # if copy == true,we just edit once
    if copy:
        model = deepcopy(model)
    device = torch.device(f'cuda:{hparams.device}')
    editor = GRACE(model=model, config=hparams,device=device)

    tokens = tokenize(request,tokenizer=tok,device=device)
    editor.edit(config=hparams,tokens=tokens)


    return editor,{}


