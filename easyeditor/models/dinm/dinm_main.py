from copy import deepcopy
from typing import Any, Dict, List, Tuple
from collections import deque

import torch
from torch.nn import CrossEntropyLoss
from transformers import AutoModelForCausalLM, AutoTokenizer

from ...util import nethook

from .dinm_hparams import DINMHyperParams
from ...trainer import kl_loc_loss, masked_log_probs


def apply_dinm_to_model(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    requests: List[Dict],
    hparams: DINMHyperParams,
    copy=False,
    return_orig_weights=False,
    keep_original_weight=False,
    **kwargs: Any,
) -> Tuple[AutoModelForCausalLM, Dict[str, Any]]:
    """
    Returns a model with the desired changes.
    :param copy: If true, will preserve the original model while creating a new one to edit.
        Note that you are responsible for deallocating the new model's memory to avoid leaks.
    :return: (1) the updated model, (2) the weights that changed
    """
    weights_copy = {}
    if copy:
        model = deepcopy(model)

    deltas = execute_dinm(model, tok, requests, hparams)

    with torch.no_grad():
        for w_name, upd_matrix in deltas.items():
            w = nethook.get_parameter(model, w_name)
            if return_orig_weights and w_name not in weights_copy:
                weights_copy[w_name] = w.detach().clone()

            w[...] += upd_matrix

    print(f"New weights successfully inserted into {list(deltas.keys())}")

    if not keep_original_weight:
        weights_copy = {}

    return model, weights_copy


def get_edit_labels(tok, labels):
    return labels.masked_fill(labels == tok.pad_token_id, -100)




def execute_dinm(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    requests: List[Dict],
    hparams: DINMHyperParams,
    **kwargs: Any,
) -> Dict[str, Tuple[torch.Tensor]]:
    """
    Executes the FT update algorithm for the specified update at the specified layer
    Invariant: model at beginning of function == model at end of function
    """
    device = torch.device(f'cuda:{hparams.device}')
    # model = model.to(device)
    # Update target and print info
    requests = deepcopy(requests)
    for request in requests:
        if request["target_new"] != " ":
            # Space required for correct tokenization
            request["target_new"] = " " + request["target_new"]
        print(
            f"Executing FT algo for: "
            f"[{request['prompt']}] -> [{request['target_new']}]"
        )

    
    # Retrieve weights that user desires to change
    weights = {
        n: p
        for n, p in model.named_parameters()
        for layer in hparams.layers   # specific layer for each instance
        if hparams.rewrite_module_tmp.format(layer) in n
    }
    
    # Save old weights for future restoration
    weights_copy = {k: v.detach().clone() for k, v in weights.items()}
    print(f"Weights to be updated: {list(weights.keys())}")

    # Configure optimizer / gradients
    opt = torch.optim.Adam(
        [v for _, v in weights.items()],
        lr=hparams.lr,
        weight_decay=hparams.weight_decay,
    )
    for name, w in model.named_parameters():
        w.requires_grad = name in weights


    ######## general knowledge constraint#####################
    instruction_TextsandTargets = [r["locality"]["general knowledge constraint"]["prompt"] + " " + r["locality"]["general knowledge constraint"]["ground_truth"] for r in requests]
    with torch.no_grad():
            instructandAns = dict(
                tok(
                    instruction_TextsandTargets,
                    return_tensors="pt", padding=True, truncation=True
                ).to(device)   #  torch.Size([1, 148])
            )
            instructonlyAns = dict(
                tok(
                    [r["locality"]["general knowledge constraint"]["ground_truth"] for r in requests],
                    return_tensors="pt", padding=True, truncation=True
                ).to(device)  
            )  #  torch.Size([1, 59])
    instruction_base_Logits = model(**instructandAns).logits  # (B, L, D) (1,148,32000)
    instruction_base_Logits = instruction_base_Logits[:, -instructonlyAns["attention_mask"].size(1):]  #torch.Size([1, 59, 32000])
    
    ############edit toxic regions#############################
    # # Update loop: intervene at layers simultaneously
    # loss_meter = AverageMeter()
    ft_input = [request["prompt"] + " " + request["target_new"] for request in requests]
    out_ids = dict(tok(request["target_new"], return_tensors="pt", padding=True).to(device))  #torch.Size([1, 69])
    out_labels = get_edit_labels(tok, out_ids["input_ids"])

    for it in range(hparams.num_steps):
        print(20 * "=")
        print(f"Epoch: {it}")
        print(20 * "=")
        inputs = tok(ft_input, return_tensors="pt", padding=True).to(device)
        opt.zero_grad()
        output = model(**inputs).logits  #torch.Size([1, 321, 32000])
        loss_dict = masked_log_probs(hparams, output, out_labels, shift=True)
        l_edit = loss_dict["nll"]
        with torch.no_grad():
            post_logits = model(**instructandAns).logits  # (B, L, D) tensor (1,59,32000)
        kl_mask = instructonlyAns["attention_mask"]
        if kl_mask.size(1) != post_logits.size(1):  #torch.Size([1, 59, 32000])
            post_logits = post_logits[:, -kl_mask.size(1):]   #torch.Size([1, 59, 32000])
        l_loc_instruction = kl_loc_loss(instruction_base_Logits.detach(), post_logits, mask=kl_mask) # tensor 一个值 0
        loss = hparams.kl_factor  * l_edit + l_loc_instruction
        # loss =  l_edit 
        print(f"Batch loss {loss.item()}, loss_edit*0.1:{0.1 * l_edit}, loss_loc_instruction:{l_loc_instruction}")

        if loss.item() >= 1e-4:
            loss.backward()
            opt.step()
            

            if type(hparams.norm_constraint) is float:
                eps = hparams.norm_constraint
                with torch.no_grad():
                    for k, v in weights.items():
                        v[...] = torch.clamp(
                            v, min=weights_copy[k] - eps, max=weights_copy[k] + eps
                        )
        else:
            break

    deltas = {k: (weights[k] - weights_copy[k]).detach() for k in weights}

    # Restore state of original model
    with torch.no_grad():
        for k, v in weights.items():
            v[...] = weights_copy[k]

    print(f"Deltas successfully computed for {list(weights.keys())}")

    return deltas



def chunks(arr, n):
    """Yield successive n-sized chunks from arr."""
    chunk = []
    for a in arr:
        chunk.append(a)
        if len(chunk) == n:
            yield chunk
            chunk = []
    if len(chunk) > 0:
        yield chunk


class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
