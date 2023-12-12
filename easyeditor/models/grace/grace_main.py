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

    device = torch.device(f'cuda:{hparams.device}')
    editor = GRACE(model=model, config=hparams,device=device)

    for request in requests:
        print(
            f"Executing GRACE algo for: "
            f"[{request['prompt']}] -> [{request['target_new']}]"
        )
        tokens = tokenize(request,tokenizer=tok,device=device)
        editor.edit(config=hparams,tokens=tokens)

    return editor,{}


