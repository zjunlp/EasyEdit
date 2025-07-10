import copy
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from .compute_z import compute_z
from torch.optim.lr_scheduler import CosineAnnealingLR
from util import nethook
from util.globals import *
import torch.optim as optim
from util.layer_stats import layer_stats
import argparse

import numpy as np
import os
from transformers.modeling_attn_mask_utils import AttentionMaskConverter,_prepare_4d_causal_attention_mask
from .AlphaEdit_ARE_hparams import AlphaEditAREHyperParams
COV_CACHE = {}
def compute_ks(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    batch_data: list,
    hparams: AlphaEditAREHyperParams,
    layer: int,
    idxs_dict:dict,
):
    input_ids = tok(batch_data, padding=True,return_tensors="pt").to("cuda")
    zs_out_dict = {}

    with torch.no_grad():
        with nethook.Trace(
            module=model,
            layer=hparams.layer_module_tmp.format(layer),
            retain_input=True,
            retain_output=True,
            detach=True,
            clone=True,
            ) as tr:
                _ = model(**input_ids)
                #layer_in_ks = tr.input #(bs:seq:h_dim)
                zs_out = tr.output#(bs:seq:h_dim)
    zs_out = zs_out[0] if type(zs_out) is tuple else zs_out
    zs_out_list = []
    for k, idxs in idxs_dict.items():
        for idx in idxs:
            zs_out_list.append(zs_out[k,idx])
    zs_out = torch.stack(zs_out_list, dim=1)
    return zs_out


def apply_AlphaEdit_ARE_to_model(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    hparams:AlphaEditAREHyperParams,
    batch_data:list,
    P = None,):

    weights = {
        f"{hparams.rewrite_module_tmp.format(layer)}.weight": nethook.get_parameter(
            model, f"{hparams.rewrite_module_tmp.format(layer)}.weight"
        )
        for layer in hparams.layers
    }
    # Save old weights for future restoration
    weights_copy = {k: v.detach().clone() for k, v in weights.items()}




    z_layer = hparams.layers[-1]
    all_zs_list = []
    idxs_dict = {}
    for k, data in enumerate(batch_data):
        idxs_list, zs_list = compute_z(
            model,
            tok,
            data,
            z_layer,
            hparams
        )
        all_zs_list.extend(zs_list)
        idxs_dict[k] = idxs_list
    zs = torch.stack(all_zs_list, dim = 1)
    batch_question_ans = [
        i['question'] + i['answer'] for i in batch_data
    ]
    
    # Insert
    for i, layer in enumerate(hparams.layers):
        #print(f"\n\nLAYER {layer}\n")
        contexts_tok = tok(batch_question_ans, padding=True, return_tensors="pt").to(
            next(model.parameters()).device
        )
        with torch.no_grad():
            with nethook.Trace(
                module=model,
                layer=hparams.rewrite_module_tmp.format(layer),
                retain_input=True,
                retain_output=True,
                detach=True,
                clone=True,
            ) as tr:
                _ = model(**contexts_tok)
                layer_in_ks = tr.input #(bs:seq:h_dim)
                layer_out_ks = tr.output#(bs:seq:h_dim)
        layer_out_ks = layer_out_ks[0] if type(layer_out_ks) is tuple else layer_out_ks
        

        cur_zs = compute_ks(model, tok,batch_question_ans, hparams, z_layer, idxs_dict)
        targets = zs - cur_zs
        print("z error", torch.linalg.norm(targets, dim=0).mean())
        # ex_tok = tok(ex_data, padding=True, return_tensors="pt").to(
        #     next(model.parameters()).device
        # )
        ks_list = []
        kp_list = []
        for k, idxs in idxs_dict.items():
            all_idxs = set(range(len(layer_in_ks[k])))
            unselected_idxs = list(all_idxs - set(idxs))
            for idx in idxs:
                ks_list.append(layer_in_ks[k, idx])
            for unselected_idx in unselected_idxs:
                kp_list.append(layer_in_ks[k, unselected_idx])
        layer_ks = torch.stack(ks_list, dim=1)
        layer_kp = torch.stack(kp_list, dim=1)


        resid = targets / (len(hparams.layers) - i)  # Distribute residual across layers
        upd_matrix = torch.linalg.solve(
                P[i,:,:].cuda() @ (layer_ks @ layer_ks.T + layer_kp @ layer_kp.T)  + hparams.L2*torch.eye(layer_ks.shape[0], dtype=torch.float,device="cuda"), P[i,:,:].cuda() @ layer_ks @ resid.T
        )
        # Adjust update matrix shape
        weight_name = f"{hparams.rewrite_module_tmp.format(layer)}.weight"
        upd_matrix = upd_matrix_match_shape(upd_matrix, weights[weight_name].shape)
        print("orig norm", torch.linalg.norm(weights[weight_name]))
        print("upd norm", torch.linalg.norm(upd_matrix))

        # Update model weights and record desired changes in `delta` variable
        with torch.no_grad():
            weights[weight_name][...] = weights_copy[weight_name] + upd_matrix.float()
        # Clear GPU memory
        for x in [layer_ks,layer_kp, cur_zs, targets, layer_in_ks, layer_out_ks,P]:
            x.cpu()
            del x
        torch.cuda.empty_cache()
    return weights_copy

def get_cov(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    layer_name: str,
    mom2_dataset: str,
    mom2_n_samples: str,
    mom2_dtype: str,
    inv: bool = False,
    force_recompute: bool = False,
) -> torch.Tensor:
    """
    Retrieves covariance statistics, then computes the algebraic inverse.
    Caches result for future use.
    """

    model_name = model.config._name_or_path.replace("/", "_")
    key = (model_name, layer_name)

    print(f"Retrieving covariance statistics for {model_name} @ {layer_name}.")
    if key not in COV_CACHE or force_recompute:
        stat = layer_stats(
            model,
            tok,
            layer_name,
            STATS_DIR,
            mom2_dataset,
            to_collect=["mom2"],
            sample_size=mom2_n_samples,
            precision=mom2_dtype,
            force_recompute=force_recompute,
        )
        COV_CACHE[key] = stat.mom2.moment().float().to("cpu")

    return (
        torch.inverse(COV_CACHE[key].to("cuda")) if inv else COV_CACHE[key].to("cuda")
    )


def upd_matrix_match_shape(matrix: torch.Tensor, shape: torch.Size) -> torch.Tensor:
    """
    GPT-2 and GPT-J have transposed weight representations.
    Returns a matrix that matches the desired shape, else raises a ValueError
    """

    if matrix.shape == shape:
        return matrix
    elif matrix.T.shape == shape:
        return matrix.T
    else:
        raise ValueError(
            "Update matrix computed by MEMIT does not match original weight shape. "
            "Check for bugs in the code?"
        )
