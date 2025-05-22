from typing import Dict, List

import numpy as np
import time
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .compute_z import get_module_input_output_at_words
from .hparams import NAMETHyperParams

import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

def chunks(arr, n):
    for i in range(0, len(arr), n):
        yield arr[i:i+n]

def compute_ks(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    requests: Dict,
    hparams: NAMETHyperParams,
    layer: int,
    context_templates: List[str],
):

    all_temps = [
        context.format(request["prompt"])
        for request in requests
        for context_type in context_templates
        for context in context_type
    ]
    trigger_words = [
        request["subject"]
        for request in requests
        for context_type in context_templates
        for _ in context_type
    ]

    layer_ks, _ = get_module_input_output_at_words(
        model,
        tok,
        layer,
        context_templates=all_temps,
        words=trigger_words,
        module_template=hparams.rewrite_module_tmp,
        fact_token_strategy=hparams.fact_token,
        change_padding=True
    )

    print(f"inside layer_ks, size:{layer_ks.size()}")
    context_type_lens = [0] + [len(context_type) for context_type in context_templates]
    context_len = sum(context_type_lens)
    context_type_csum = np.cumsum(context_type_lens).tolist()
    ans = []

    for i in range(0, layer_ks.size(0), context_len): #for one context
        tmp = []  #layer_ks[i:i+context_len]

        for j in range(len(context_type_csum) - 1):
            start, end = context_type_csum[j], context_type_csum[j + 1]
            tmp.append(layer_ks[i + start : i + end].mean(0)) #mean for reprs for the same context
        ans.append(torch.stack(tmp, 0).mean(0)) #mean for reprs for different contexts

    return torch.stack(ans, dim=0) #stack for different request, a total of 4 for each batch
