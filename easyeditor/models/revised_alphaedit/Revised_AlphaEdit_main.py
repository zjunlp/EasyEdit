import os
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from ..rome.layer_stats import layer_stats
from ...util import nethook
from ...util.globals import *

from .compute_ks import compute_ks
from .compute_z import compute_z
from .Revised_AlphaEdit_hparams import Revised_AlphaEditHyperParams

from tqdm.auto import tqdm

# Cache variable(s)
CONTEXT_TEMPLATES_CACHE = None
COV_CACHE = {}

P_loaded = False
cache_c_new = False

edit_count = 0

def apply_Revised_AlphaEdit_to_model(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    requests: List[Dict],
    hparams: Revised_AlphaEditHyperParams,
    copy=False,
    return_orig_weights=False,
    cache_template: Optional[str] = None,
    keep_original_weight=False,
    **kwargs
) -> Dict[str, Tuple[torch.Tensor]]:
  #-> Tuple[AutoModelForCausalLM, Dict[str, Any]]:
    """
    Returns a model with the desired changes.
    :param copy: If true, will preserve the original model while creating a new one to edit.
        Note that you are responsible for deallocating the new model's memory to avoid leaks.
    :return: (1) the updated model, (2) an original copy of the weights that changed
    """

    global P, P_loaded, cache_c, cache_c_new
    
    weights_copy = {}
    if copy:
        model = deepcopy(model)
    
    # Calculate the null-space projection matrix P
    # Please ensure that you have downloaded "null_space_project.pt" to the easyedit folder beforehand, or get the P by following calculation
    if not os.path.exists(hparams.P_loc):
        print(f"The null-space projection matrix P does not exist and now calculate.")
        P = {}
        for layer in tqdm(hparams.layers, desc="Computing projection matrix"):
            P[layer] = get_project(model, tok, layer, hparams).to("cpu")
        print("Saving null-space projection matrix P to avoid redundant future computations...")
        torch.save(P, hparams.P_loc)
        P_loaded = True
    elif P_loaded == False:
        P = torch.load(hparams.P_loc)
        P = {k: v.contiguous() for k, v in P.items()}
        P_loaded = True

    # Maintain the global variable cache_c to avoid redundant computations.
    # If this is the first calculation (i.e., cache_c_new == false), then initialize cache_c first
    if not cache_c_new:
        W_out = nethook.get_parameter(model, f"{hparams.rewrite_module_tmp.format(0)}.weight")
        if "gpt2-xl" in hparams.model_name.lower():
            cache_c_shape = (W_out.shape[0], W_out.shape[0])
        else: # modern LLMs
            cache_c_shape = (W_out.shape[1], W_out.shape[1])

        cache_c = {
            layer: torch.zeros(cache_c_shape) for layer in hparams.layers
        }
        del W_out
        cache_c_new = True
    
    return_statistics = kwargs.pop("return_statistics", False)
    execution_result = execute_Revised_AlphaEdit(
        model,
        tok,
        requests,
        hparams,
        cache_template=cache_template,
        return_statistics=return_statistics,
    )
    if return_statistics:
        deltas, statistics = execution_result
    else:
        deltas = execution_result

    with torch.no_grad():
        for w_name, upd_m in deltas.items():
            w = nethook.get_parameter(model, w_name)
            upd_matrix = upd_m.to(w.device)
            upd_matrix = upd_matrix_match_shape(upd_matrix, w.shape)

            if return_orig_weights and w_name not in weights_copy:
                weights_copy[w_name] = w.detach().clone()
            w[...] += upd_matrix.float()
            del upd_matrix

    if len(deltas) == 0:
        print("No updates were made for these requests.")
    else:
        print(f"New weights successfully inserted into {list(deltas.keys())}")

    if return_statistics:
        return model, weights_copy, statistics
    return model, weights_copy


def execute_Revised_AlphaEdit(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    requests: List[Dict],
    hparams: Revised_AlphaEditHyperParams,
    cache_template: Optional[str] = None,
    return_statistics: bool = False,
) -> Dict[str, Tuple[torch.Tensor]] | Tuple[Dict[str, Tuple[torch.Tensor]], List[Dict[str, Any]]]:
    global edit_count

    deltas = {}
    statistics = []

    # Update target and print info
    requests = deepcopy(requests)
    for i, request in enumerate(requests):
        if '{}' not in request['prompt']:
            assert request['subject'] in request['prompt'] or \
                   print(f"Subject: {request['subject']} do not exist in prompt: {request['prompt']}")
        else:
            requests[i]['prompt'] = request['prompt'].replace('{}', request['subject'])

    # Retrieve weights that user desires to change
    weights = {
        f"{hparams.rewrite_module_tmp.format(layer)}.weight": nethook.get_parameter(
            model, f"{hparams.rewrite_module_tmp.format(layer)}.weight"
        )
        for layer in hparams.layers
    }

    # Save old weights for future restoration
    weights_copy = {k: v.detach().clone() for k, v in weights.items()}

    # Compute z for final layer
    context_templates = get_context_templates(model=model, tokenizer=tok)
    print(f"Context templates used for computing z and k/v pairs: {context_templates}")

    for request in requests:
        print(
            f"Executing AlphaEdit_Circuit algo\n"
            f"Request: [{request['prompt']}] [{request['target_true']}] -> [{request['target_new']}]"
        )

        total_source_updates = len(hparams.layers)
        updates_done = 0
        for layer in hparams.layers: # ablation
            z_list = []
            results = compute_z(
                model=model,
                tok=tok,
                request=request,
                hparams=hparams,
                layer=hparams.layers[-1],
                context_templates=context_templates,
                return_statistics=return_statistics,
            )
            cur_z = results["cur_z"]
            target_init = results["target_init"]
            if return_statistics:
                target_statistics = results["target_statistics"]

            if target_init.norm() > 1000:
                print(f"Warning: target_init norm is {target_init.norm()}, which is unusually large. This may indicate an issue with the computation.")
                break
            
            z_list.append(cur_z)
            zs = torch.stack(z_list, dim=1) # shape [d_model, num_requests]
            
            # Get current model activations
            layer_ks = compute_ks(
                model=model,
                tok=tok,
                requests=[request],
                hparams=hparams,
                layer=layer,
                context_templates=context_templates,
            ).T # shape [mlp_hidden_size, num_requests]
            print(f"Writing {layer_ks.size(1)} key/value pair(s) into layer {layer}")

            # Compute residual error
            all_prompts = []
            all_prompts.append(request["prompt"])
            
            tok.padding_side = "right"
            cur_zs_tok = tok(
                all_prompts,
                return_tensors="pt",
                padding=True,
                add_special_tokens=True,
            ).to(model.get_input_embeddings().weight.device)
            
            idxs = []
            if hparams.fact_token == "subject_first":
                start_char = all_prompts[0].find(request["subject"])
                start_tok = cur_zs_tok.char_to_token(0, start_char)
                idxs.append(start_tok)
            elif hparams.fact_token == "subject_last":
                start_char = all_prompts[0].find(request["subject"])
                start_tok = cur_zs_tok.char_to_token(0, start_char)
                end_char = start_char + len(request["subject"]) - 1
                end_tok = cur_zs_tok.char_to_token(0, end_char)
                idxs.append(end_tok)
            print(f"cur_zs_idxs: {idxs}")
            for i, idx in enumerate(idxs):
                print(f"Prompt {i}: {tok.decode(cur_zs_tok['input_ids'][i][:idx + 1])}")
            
            with torch.no_grad():
                with nethook.Trace(
                    module=model,
                    layer=hparams.layer_module_tmp.format(hparams.layers[-1]),
                    retain_output=True,
                    stop=True,
                ) as tr:
                    model(**cur_zs_tok)
                    del cur_zs_tok
            
            cur_zs = tr.output[list(range(tr.output.shape[0])), idxs].T # shape [d_model, num_requests]
            targets = zs - cur_zs
            del zs, cur_zs
            z_error = torch.linalg.norm(targets, dim=0).mean()
            print("z error", z_error)

            repeat_factor = layer_ks.size(1) // targets.size(1)
            targets = targets.repeat_interleave(repeat_factor, dim=1)
            resid = targets / (total_source_updates - updates_done)  # Distribute residual across remaining hub/source edits
            del targets
            weight_name = f"{hparams.rewrite_module_tmp.format(layer)}.weight"
            layer_device = weights[weight_name].device
            proj = P[layer].to(device=layer_device, dtype=torch.float).detach()
            layer_ks = layer_ks.to(device=layer_device, dtype=torch.float).detach()
            resid = resid.to(device=layer_device, dtype=torch.float).detach()
            c = cache_c[layer].to(device=layer_device, dtype=torch.float, copy=True).detach()

            with torch.no_grad():
                k1k1 = layer_ks @ layer_ks.T
                cache_c[layer] += (k1k1).to(cache_c[layer].device)
                k1k1.add_(c)
                del c

                lhs = proj @ k1k1
                del k1k1

                lhs.diagonal().add_(hparams.L2)

                projected_ks = proj @ layer_ks
                rhs = projected_ks @ resid.T
                del projected_ks, proj, layer_ks, resid

                upd_matrix = torch.linalg.solve(lhs, rhs)
                del lhs, rhs

            # Adjust update matrix shape
            upd_matrix = upd_matrix_match_shape(upd_matrix, weights[weight_name].shape)

            original_weight_norm = torch.linalg.norm(weights[weight_name])
            update_norm = torch.linalg.norm(upd_matrix)
            print("orig norm", original_weight_norm)
            print("upd norm", update_norm)

            if return_statistics:
                target_statistics.update(
                    {
                        "case_id": request.get("case_id"),
                        "edit_id": edit_count,
                        "edited_layer": layer,
                        "target_layer": hparams.layers[-1],
                        "z_error": float(z_error.detach().cpu()),
                        "weight_norm_before_update": float(original_weight_norm.detach().cpu()),
                        "weight_update_norm": float(update_norm.detach().cpu()),
                    }
                )
                statistics.append(target_statistics)

            # Update model weights and record desired changes in `delta` variable
            with torch.no_grad():
                weights[weight_name][...] = weights[weight_name] + upd_matrix.float()
                if deltas.get(weight_name) is None:
                    deltas[weight_name] = upd_matrix.detach()
                else:
                    deltas[weight_name] += upd_matrix.detach()
                del upd_matrix

            updates_done += 1

    # Restore state of original model
    with torch.no_grad():
        for k, v in weights.items():
            v[...] = weights_copy[k]

    edit_count += 1
    
    if return_statistics:
        return deltas, statistics
    return deltas


def get_cov(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    layer_name: str,
    mom2_dataset: str,
    mom2_n_samples: str,
    mom2_dtype: str,
    mom2_batch_tokens: int = None,
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
            batch_tokens=mom2_batch_tokens,
            hparams=hparams,
            force_recompute=force_recompute,
        )
        COV_CACHE[key] = stat.mom2.moment().float()

    return (
        torch.inverse(COV_CACHE[key].to("cuda")) if inv else COV_CACHE[key].to(f"cuda")
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
            "Update matrix computed by AlphaEdit does not match original weight shape. "
            "Check for bugs in the code?"
        )


def get_context_templates(model, tokenizer):
    global CONTEXT_TEMPLATES_CACHE

    if CONTEXT_TEMPLATES_CACHE is None:
        CONTEXT_TEMPLATES_CACHE = [["{prompt}"]]
        prompt_tok = tokenizer(
            ["The", "Therefore", "Because", "I", "You"],
            padding=True,
            return_tensors="pt",
        ).to(model.get_input_embeddings().weight.device)
        for length, n_gen in [(10, 5)]:
            gen_token = model.generate(
                **prompt_tok,
                max_new_tokens=length,
                num_beams=n_gen // 5,
                num_return_sequences=n_gen // 5,
                pad_token_id=tokenizer.eos_token_id,
            )
            templates = tokenizer.batch_decode(gen_token, skip_special_tokens=True)
            templates = [f"{template}. {{prompt}}" for template in templates]
            CONTEXT_TEMPLATES_CACHE.append(templates)

    return CONTEXT_TEMPLATES_CACHE

def get_project(model, tok, layer, hparams):
    """
    Caveat: Compute with Eigendecomposition instead of Singular Value Decomposition for efficiency because covariance is positive semi-definite.
    """
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
        hparams.mom2_batch_tokens,
        force_recompute=force_recompute,
        hparams=hparams
    )
    print(f"Computing projection matrix for layer {layer}")
    # if cov shape is (mlp_hidden_size, mlp_hidden_size), then
    #   U shape is (mlp_hidden_size, mlp_hidden_size)
    #   S shape is (mlp_hidden_size,)
    #   U[:, small_singular_indices] shape is (mlp_hidden_size, num_small_singular_values)
    #   P shape is (mlp_hidden_size, mlp_hidden_size)
    vals, vecs = torch.linalg.eigh(cov)
    threshold = hparams.nullspace_threshold
    small_singular_indices = (vals < threshold).nonzero(as_tuple=True)[0]
    print(f"{len(small_singular_indices)} small singular values found below threshold {threshold} out of {len(vals)} total singular values.")
    return vecs[:, small_singular_indices] @ vecs[:, small_singular_indices].T
