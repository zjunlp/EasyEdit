import os
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from itertools import chain

import numpy as np
import torch
import copy
import time
import random
from tqdm import *
from torch.utils.data import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from ..rome.layer_stats import layer_stats
from ...util import nethook
from ...util.generate import generate_fast, generate_standard
from ...util.globals import *

from .compute_ks import compute_ks
from .compute_z import compute_z, get_module_input_output_at_words, find_fact_lookup_idx
from .hparams import NAMETHyperParams

# Cache variable(s)
CONTEXT_TEMPLATES_CACHE = None
COV_CACHE = {}

def apply_namet_to_model(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    requests: List[Dict],
    hparams: NAMETHyperParams,
    copy=False,
    return_orig_weights=False,
    cache_template: Optional[str] = None,
    cache_id: int = 0,
) -> Tuple[AutoModelForCausalLM, Dict[str, Any]]:
    """
    Returns a model with the desired changes.
    :param copy: If true, will preserve the original model while creating a new one to edit.
        Note that you are responsible for deallocating the new model's memory to avoid leaks.
    :return: (1) the updated model, (2) an original copy of the weights that changed
    """
    tok.padding_side = "left"
    print(f"changing padding_side to left")
    def chunks(ds, n):
        for i in range(0, len(ds), n):
            yield ds[i:i + n]

    weights_copy = {}
    if copy:
        model = deepcopy(model)

    # requests = requests + list(chain(*[record for record in cali_chunks]))

    deltas = execute_namet(model, tok, requests, hparams, 
                            cache_template=cache_template, 
                            cache_id=cache_id)

    with torch.no_grad():
        for w_name, (key_mat, val_mat) in deltas.items():
            key_mat, val_mat = key_mat.to("cuda"), val_mat.to("cuda")
            upd_matrix = key_mat @ val_mat.T
            w = nethook.get_parameter(model, w_name)
            upd_matrix = upd_matrix_match_shape(upd_matrix, w.shape)

            if return_orig_weights and w_name not in weights_copy:
                weights_copy[w_name] = w.detach().clone()

            w[...] += upd_matrix.float()

    print(f"New weights successfully inserted into {list(deltas.keys())}")
    tok.padding_side = "right"
    print(f"restoring padding_side to right")
    return model, weights_copy


def execute_namet(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    requests: List[Dict],
    hparams: NAMETHyperParams,
    cache_template: Optional[str] = None,
    cache_id: int = 0
) -> Dict[str, Tuple[torch.Tensor]]:
    """
    Executes the MEMIT update algorithm for the specified update at the specified layer
    Invariant: model at beginning of function == model at end of function
    """
    deltas = {}

    requests = deepcopy(requests)

    for i, request in enumerate(requests):
        if request["target_new"]["str"][0] != " ":
            requests[i]["target_new"]["str"] = " " + request["target_new"]["str"]

    weights = {
        f"{hparams.rewrite_module_tmp.format(layer)}.weight": nethook.get_parameter(
            model, f"{hparams.rewrite_module_tmp.format(layer)}.weight"
        )
        for layer in hparams.layers
    }
    # Save old weights for future restoration
    # k is the name of the param, and v is the corresponding value
    weights_copy = {k: v.detach().clone() for k, v in weights.items()}

    # Compute z for final layer

    z_layer = hparams.layers[-1]
    opt_zs_list = []
    target_zs_list = []
    context_templates = get_context_templates(model, tok)

    print(f"Start optimizing z for requests:")
    print(f"")
    for re_id, request in enumerate(requests):
        if re_id % 10 == 0:
            print(f"optimize for {re_id}th request")
        # Retrieve k/v pair if already stored in cache
        if cache_id==0:
            cache_fname = (
                Path(
                    str(cache_template).format(
                        hparams.v_lr, z_layer, str(request["case_id"])
                    )
                )
                if cache_template is not None
                else None
            )
        else:
            cache_fname = (
                Path(
                    str(cache_template).format(
                        hparams.v_lr, z_layer, str(request["case_id"]), cache_id
                    )
                )
                if cache_template is not None
                else None
            )

        data_loaded = False
        if (
            cache_fname is not None  # Require cache template
            and cache_fname.exists()  # Cache file must exist
        ):
            try:
                data = np.load(cache_fname)
                opt_zs_list.append(torch.from_numpy(data["v_star"]).to("cuda"))
                data_loaded = True
            except Exception as e:
                print(f"Error reading cache file due to {e}. Recomputing...")

        # Compute k/v pair if not loaded from cache
        if not data_loaded:
            opt_zs = compute_z(
                model,
                tok,
                request,
                hparams,
                z_layer,
                context_templates
            )

            if cache_fname is not None:
                try:
                    cache_fname.parent.mkdir(exist_ok=True, parents=True)
                    np.savez(
                        cache_fname,
                        **{
                            "v_star": opt_zs.detach().cpu().numpy(),
                        },
                    )
                    print(f"Cached k/v pair at {cache_fname}")
                except Exception as e:
                    print(f"Error loading cache file due to {e}.")

            opt_zs_list.append(opt_zs)

    # Insert
    edit_layers = hparams.layers
    for i, layer in enumerate(edit_layers):
        print(f"\n\nLAYER {layer}\n")

        # Get current model activations
        layer_ks = compute_ks(model, tok, requests, hparams, layer, context_templates).T
        print(f"size of layer_ks:{layer_ks.size()}")

        target_zs_list=[]
        for re_id, request in enumerate(requests):
            temp_templates = context_templates + [["{} is a"]]
            all_temps = [
                context.format(request["prompt"])
                for context_type in temp_templates
                for context in context_type
            ]
            all_temp_words = [request["subject"] for _ in all_temps]
            cur_zs = get_module_input_output_at_words(
                model,
                tok,
                z_layer,
                context_templates=all_temps,
                words=all_temp_words,
                module_template=hparams.layer_module_tmp,
                fact_token_strategy=hparams.fact_token,
            )[1].T #0 for the module input before layer_module, 1 for the output after layer_module
            target_zs = opt_zs_list[re_id][:,1:-1] - cur_zs[:,1:-1]
            temp_target_zs = torch.mean(target_zs, dim=1)
            target_zs_list.append(temp_target_zs)

        targets = torch.stack(target_zs_list, dim=1) # targets to be distributed across layers

        # after transpose, layer_ks.size(1) and targets.size(1) means the number
        # of entries
        repeat_factor = (layer_ks.size(1) // targets.size(1)) 
        targets = targets.repeat_interleave(repeat_factor, dim=1)
        # Load covariance matrix
        force_recompute = False
        cov = get_cov(
            model,
            tok,
            hparams.rewrite_module_tmp.format(layer),
            hparams.mom2_dataset,
            hparams.mom2_n_samples
            if not force_recompute
            else hparams.mom2_n_samples // 10,
            hparams.mom2_dtype,
            force_recompute=force_recompute,
            hparams=hparams
        )

        # Compute update in double precision
        if torch.cuda.device_count() == 1:
            layer_ks, targets = (
                layer_ks.double(),
                targets.double(),
            )
        else:
            layer_ks, targets = (
                layer_ks.double().to("cuda:1"),
                targets.double().to("cuda:1"),
            )

        cov_mat = hparams.mom2_update_weight[i] * cov.double() + (layer_ks @ layer_ks.T)
        start_time = time.time()
        if torch.cuda.device_count() == 1:
            adj_k = torch.inverse(cov_mat.to("cpu")).to("cuda") \
                @ layer_ks
        else:
            adj_k = torch.inverse(cov_mat.to("cuda:1")).to("cuda:1") \
                @ layer_ks
        print(f"computing inverse takes:{time.time()-start_time}")
        resid = targets / (len(edit_layers) - i) # Distribute residual across layers
        upd_matrix = resid @ adj_k.T

        # Adjust update matrix shape
        weight_name = f"{hparams.rewrite_module_tmp.format(layer)}.weight"
        upd_matrix = upd_matrix_match_shape(upd_matrix, weights[weight_name].shape)
        scaled_cov = hparams.mom2_update_weight[i] * cov.double()
        failure_norm = upd_matrix @ scaled_cov
        locality_norm = upd_matrix @ (layer_ks @ layer_ks.T)
        target_key = targets @ layer_ks.T

        print(f"failure norm for {layer}th layer:{torch.norm(failure_norm)}")
        print(f"locality norm for {layer}th layer:{torch.norm(locality_norm)}")
        print(f"target@key norm for {layer}th layer:{torch.norm(target_key)}")
        print(f"layer_ks@layer_ks.T norm:{torch.norm((layer_ks @ layer_ks.T))}")
        print(f"scaled_cov norm:{torch.norm(scaled_cov)}")
        print(f"upd_matrix norm:{torch.norm(upd_matrix)}")

        # Update model weights and record desired changes in `delta` variable
        with torch.no_grad():
            if torch.cuda.device_count() == 1:
                weights[weight_name][...] = weights_copy[weight_name] + upd_matrix.float()
            else:
                weights[weight_name][...] = weights_copy[weight_name] + upd_matrix.float().to("cuda:0")
            #weights[weight_name][...] = weights_copy[weight_name] + upd_matrix.float()
            deltas[weight_name] = (
                adj_k.detach().cpu(),
                resid.detach().cpu(),
            )

        # Clear GPU memory
        cov.cpu()
        for x in [layer_ks, targets]:
            x.cpu()
            del x
        torch.cuda.empty_cache()

    # Restore state of original model
    with torch.no_grad():
        for k, _ in weights.items():
            nethook.get_parameter(model, k)[...] = weights_copy[k]

    # with torch.no_grad():
    #     for k, v in weights.items():
    #         v[...] = weights_copy[k]

    print(f"Deltas successfully computed for {list(weights.keys())}")

    return deltas


def get_cov(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    layer_name: str,
    mom2_dataset: str,
    mom2_n_samples: str,
    mom2_dtype: str,
    inv: bool = False,
    force_recompute: bool = False,
    hparams=None,
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
            hparams.stats_dir,
            mom2_dataset,
            to_collect=["mom2"],
            sample_size=mom2_n_samples,
            precision=mom2_dtype,
            force_recompute=force_recompute,
        )
        COV_CACHE[key] = stat.mom2.moment().float().to("cpu")

    if torch.cuda.device_count() == 1:  
        return (
            torch.inverse(COV_CACHE[key].to("cuda")) if inv else COV_CACHE[key].to("cuda")
        )
    else:
        try:
            return (
                torch.inverse(COV_CACHE[key].to("cuda:1")) if inv else COV_CACHE[key].to("cuda:1")
            )
        except:
            return (
                torch.inverse(COV_CACHE[key].to("cuda:0")) if inv else COV_CACHE[key].to("cuda:0")
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


def get_context_templates(model, tok):
    global CONTEXT_TEMPLATES_CACHE

    print(f"Generating using generate_standard")
    temperature=0.5
    top_k=100
    
    if CONTEXT_TEMPLATES_CACHE==None:
        if "deepseek" in str(model.config._name_or_path).lower():
            initial_tplt = ["The", "Therefore", "Because", "I", "You"]
        else:
            initial_tplt = ["The", "Therefore", "Because", "I", "You", \
                            "However", "Also", "Nevertheless", "He", "It", \
                        "Can", "Because"]
        CONTEXT_TEMPLATES_CACHE = [["{}"]] + [
            [
                f.replace("{", " ").replace("}", " ") + ". {}"
                for f in generate_standard(
                model,
                initial_tplt,
                tok,
                max_new_tokens=length,
                do_sample=True,
                temperature=temperature, #0.5
                top_k=top_k #100
            )
            ]
            for length, n_gen in [(10, 5)]  # Be careful about changing this.
            ]
        # print(f"context_template:{CONTEXT_TEMPLATES_CACHE}")
        print(f"temperature:{temperature}")
        print(f"top_k:{top_k}")
        print(f"len(initial_tplt):{len(initial_tplt)}")

    return CONTEXT_TEMPLATES_CACHE

# def get_context_templates(model, tok):
#     global CONTEXT_TEMPLATES_CACHE

#     print(f"Generating using generate_fast")
#     if CONTEXT_TEMPLATES_CACHE is None:
#         CONTEXT_TEMPLATES_CACHE = [["{}"]] + [
#             [
#                 f.replace("{", " ").replace("}", " ") + ". {}"
#                 for f in generate_fast(
#                     model,
#                     tok,
#                     ["The", "Therefore", "Because", "I", "You"],
#                     n_gen_per_prompt=n_gen // 5,
#                     max_out_len=length,
#                 ) # 用模型生成句子
#             ]
#             for length, n_gen in [(10, 5)]  # Be careful about changing this.
#         ]
#         print(f"Cached context templates {CONTEXT_TEMPLATES_CACHE}")

#     return CONTEXT_TEMPLATES_CACHE

