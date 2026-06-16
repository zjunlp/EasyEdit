from typing import Dict, List, Tuple
from copy import deepcopy

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from ..rome import repr_tools
from ...util import nethook
from ...util.device import get_module_device, normalize_device

from .memit_FE_hparams import MEMITFEHyperParams

def compute_z(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    request: Dict,
    hparams: MEMITFEHyperParams,
    z_layers: List[int],
    context_templates: List[str],
) -> Dict[int, torch.Tensor]:
    """
    A unified function that use FE methods to calculate targets for multiple layers.
    Calculate the first layer target by _direct_compute_z().
    Perform forward propagation for the rest of the layers.
    """

    request_cp = deepcopy(request)
    z_layers = sorted(z_layers)
    zs_dict = {layer: None for layer in z_layers}  # Store computed z for each layer here, to be returned at the end

    # Directly compute the target for the first layer
    first_layer = z_layers[0]
    print(f"Calculate target for the first layer {first_layer}")
    try:
        first_layer_z = _direct_compute_z(
            model=model,
            tok=tok,
            request=request_cp,
            hparams=hparams,
            layer=first_layer,
            context_templates=context_templates,
        )
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Error computing z for layer {first_layer} due to {e}. Returning None.")
        return None
    
    if first_layer_z is None:
        print(f"Direct computation for layer {first_layer} returned None. Returning None.")
        return None

    zs_dict[first_layer] = first_layer_z.detach().clone()


    # Forward propagation
    device = normalize_device(getattr(hparams, "device", None))
    other_layers = z_layers[1:]
    first_layer_module = hparams.layer_module_tmp.format(first_layer)
    prompt = request["prompt"].format(request["subject"])
    input_tok = tok(prompt, return_tensors="pt").to(device)

    seq_len = input_tok["input_ids"].shape[1]
    
    # Find lookup index
    lookup_idx = find_fact_lookup_idx(
        request["prompt"], 
        request["subject"], 
        tok, 
        hparams.fact_token, 
        verbose=False
    )
    
    # def edit_output_fn(cur_out, cur_layer_name):
    #     first_layer_module = hparams.layer_module_tmp.format(first_layer)
    #     # Inject the first layer z into the forward pass when we reach the first layer
    #     if cur_layer_name == first_layer_module:
    #         z_to_inject = zs_dict[first_layer]
    #         if cur_out[0].shape[0] == 1:
    #             cur_out[0][0, lookup_idx, :] = z_to_inject
    #         else:
    #             cur_out[0][lookup_idx, 0, :] = z_to_inject
    #     return cur_out

    def edit_output_fn(cur_out, cur_layer_name):
        if cur_layer_name == first_layer_module:
            layer_output = nethook.get_hidden_state(cur_out)  
            z_to_inject = zs_dict[first_layer].to(device=layer_output.device, dtype=layer_output.dtype)
            
            if layer_output.shape[0] == 1: 
                layer_output[0, lookup_idx, :] = z_to_inject
            else:
                if layer_output.shape[1] == seq_len: # [batch, seq, dim]
                    layer_output[:, lookup_idx, :] = z_to_inject
                else: # [seq, batch, dim]
                    layer_output[lookup_idx, :, :] = z_to_inject
                    
            return nethook.replace_hidden_state(cur_out, layer_output)
        return cur_out
    
    with torch.no_grad():
        with nethook.TraceDict(
            module=model,
            layers=[hparams.layer_module_tmp.format(l) for l in z_layers],  
            # Must trace ALL layers including first_layer!
            edit_output=edit_output_fn,
        ) as tr:
            model(**input_tok)
        
            # Extract z from each layer
            for layer in other_layers:
                layer_module = hparams.layer_module_tmp.format(layer)
                output = tr[layer_module].output
                c_z_raw = nethook.get_hidden_state(output)
                # c_z_raw = output[0] if isinstance(output, tuple) else output
                
                # # Extract at lookup position
                # if c_z_raw.dim() == 3:
                #     c_z = c_z_raw[0, lookup_idx, :] if c_z_raw.shape[0] == 1 else c_z_raw[lookup_idx, 0, :]
                # else:
                #     c_z = c_z_raw[lookup_idx, :]

                if c_z_raw.dim() == 3:
                    if c_z_raw.shape[1] == seq_len:  # [Batch, Seq, Dim]
                        c_z = c_z_raw[0, lookup_idx, :] if c_z_raw.shape[0] == 1 else c_z_raw[:, lookup_idx, :]
                    else:                             # [Seq, Batch, Dim]
                        c_z = c_z_raw[lookup_idx, 0, :] if c_z_raw.shape[1] == 1 else c_z_raw[lookup_idx, :, :]
                else:
                    c_z = c_z_raw[lookup_idx, :]
                
                zs_dict[layer]=c_z.detach().clone()
                print()
                print(f"Computed z for layer {layer} during forward pass.")
                print(f"Norm of z for layer {layer}: {zs_dict[layer].norm().item()}")
    
    return zs_dict

def _direct_compute_z(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    request: Dict,
    hparams: MEMITFEHyperParams,
    layer: int,
    context_templates: List[str],
) -> torch.Tensor:
    """
    Computes the value (right) vector for the rank-1 update.
    Runs a simple optimization procedure.
    """

    # Get model parameters
    lm_w, ln_f = (
        nethook.get_parameter(model, f"{hparams.lm_head_module}.weight").T,
        nethook.get_module(model, hparams.ln_f_module),
    )
    try:
        lm_b = nethook.get_parameter(model, f"{hparams.lm_head_module}.bias")
    except LookupError as _:
        lm_b = next(model.parameters()).new_zeros(model.config.vocab_size)

    print("Computing right vector (v)")
    device = normalize_device(getattr(hparams, "device", None))
    rewrite_module_name = hparams.layer_module_tmp.format(layer)
    rewrite_device = get_module_device(nethook.get_module(model, rewrite_module_name), device)

    # Tokenize target into list of int token IDs
    target_ids = tok.encode(request["target_new"], return_tensors="pt", add_special_tokens=False).to(device)[0]

    if target_ids[0] == tok.bos_token_id or target_ids[0] == tok.unk_token_id:
        target_ids = target_ids[1:]
    # Compile list of rewriting and KL x/y pairs
    rewriting_prompts, kl_prompts = [
        context.format(request["prompt"]) + tok.decode(target_ids[:-1])
        for context_types in context_templates
        for context in context_types
    ], ["{} is a"]
    all_prompts = rewriting_prompts + kl_prompts

    input_tok = tok(
        [prompt.format(request["subject"]) for prompt in all_prompts],
        return_tensors="pt",
        padding=True,
    ).to(device)

    # Compute rewriting targets
    rewriting_targets = torch.tensor(-100, device=device).repeat(
        len(rewriting_prompts), *input_tok["input_ids"].shape[1:]
    )
    for i in range(len(rewriting_prompts)):
        ex_len = input_tok["attention_mask"][i].sum()
        rewriting_targets[i, ex_len - len(target_ids) : ex_len] = target_ids

    # Compute indices of the tokens where the fact is looked up
    lookup_idxs = [
        find_fact_lookup_idx(
            prompt, request["subject"], tok, hparams.fact_token, verbose=(i == 0)
        )
        for i, prompt in enumerate(all_prompts)
    ]

    # Finalize rewrite and loss layers
    loss_layer = max(hparams.v_loss_layer, layer)
    print(f"Rewrite layer is {layer}")
    print(f"Tying optimization objective to {loss_layer}")

    # Set up an optimization over a latent vector that, when output at the
    # rewrite layer, i.e. hypothesized fact lookup location, will induce the
    # target token to be predicted at the final layer.
    if hasattr(model.config, 'n_embd'):
        delta = torch.zeros((model.config.n_embd,), requires_grad=True, device=rewrite_device)
    elif hasattr(model.config, 'hidden_size'):
        delta = torch.zeros((model.config.hidden_size,), requires_grad=True, device=rewrite_device)
    else:
        raise NotImplementedError
    target_init, kl_distr_init = None, None

    # Inserts new "delta" variable at the appropriate part of the computation
    def edit_output_fn(cur_out, cur_layer):
        nonlocal target_init

        if cur_layer == rewrite_module_name:
            layer_output = nethook.get_hidden_state(cur_out)
            delta_for_hidden = delta.to(device=layer_output.device, dtype=layer_output.dtype)

            # Store initial value of the vector of interest
            if target_init is None:
                print("Recording initial value of v*")
                # Initial value is recorded for the clean sentence
                target_init = layer_output[0, lookup_idxs[0]].detach().clone()

            # Add intervened delta
            for i, idx in enumerate(lookup_idxs):

                if len(lookup_idxs)!=layer_output.shape[0]:
                    layer_output[idx, i, :] += delta_for_hidden
                else:
                    layer_output[i, idx, :] += delta_for_hidden

            return nethook.replace_hidden_state(cur_out, layer_output)

        return cur_out

    # Optimizer
    opt = torch.optim.Adam([delta], lr=hparams.v_lr)
    nethook.set_requires_grad(False, model)

    # Execute optimization
    for it in range(hparams.v_num_grad_steps):
        opt.zero_grad()

        # Forward propagation
        with nethook.TraceDict(
            module=model,
            layers=[
                hparams.layer_module_tmp.format(loss_layer),
                hparams.layer_module_tmp.format(layer),
            ],
            retain_input=False,
            retain_output=True,
            edit_output=edit_output_fn,
        ) as tr:
            logits = model(**input_tok).logits
            # Compute distribution for KL divergence
            kl_logits = torch.stack(
                [
                    logits[i - len(kl_prompts), idx, :]
                    for i, idx in enumerate(lookup_idxs[-len(kl_prompts) :])
                ],
                dim=0,
            )
            kl_log_probs = torch.nn.functional.log_softmax(kl_logits, dim=1)
            if kl_distr_init is None:
                kl_distr_init = kl_log_probs.detach().clone()

        # Compute loss on rewriting targets

        output = nethook.get_hidden_state(tr[hparams.layer_module_tmp.format(loss_layer)].output)

        # loss_layer_out = tr[hparams.layer_module_tmp.format(loss_layer)].output
        # if isinstance(loss_layer_out, (list, tuple)):
        #     output = loss_layer_out[0]
        # else:
        #     output = loss_layer_out

        if output.shape[1]!=rewriting_targets.shape[1]:
            output=torch.transpose(output, 0, 1)
        full_repr = output[:len(rewriting_prompts)]

        full_repr = ln_f(full_repr)
        log_probs = torch.log_softmax(full_repr @ lm_w.to(full_repr.device) + lm_b.to(full_repr.device), dim=2)
        loss = torch.gather(
            log_probs,
            2,
            torch.where(rewriting_targets != -100, rewriting_targets, 0).unsqueeze(2).to(log_probs.device),
        ).squeeze(2)
        mask = (rewriting_targets != -100).float()

        # Aggregate total losses
        nll_loss_each = -(loss * mask.to(loss.device)).sum(1) / target_ids.size(0)
        nll_loss = nll_loss_each.mean()
        kl_loss = hparams.kl_factor * torch.nn.functional.kl_div(
            kl_distr_init, kl_log_probs, log_target=True, reduction="batchmean"
        )
        weight_decay = hparams.v_weight_decay * (
            torch.norm(delta) / torch.norm(target_init) ** 2
        )
        # weight_decay = hparams.v_weight_decay * torch.norm(delta) ** 2
        loss = nll_loss + kl_loss.to(nll_loss.device) + weight_decay.to(nll_loss.device)
        print(
            f"loss {np.round(loss.item(), 3)} = {np.round(nll_loss.item(), 3)} + {np.round(kl_loss.item(), 3)} + {np.round(weight_decay.item(), 3)} "
            f"avg prob of [{request['target_new']}] "
            f"{torch.exp(-nll_loss_each).mean().item()}"
        )

        if loss < 5e-2:
            break

        if it == hparams.v_num_grad_steps - 1:
            break

        # Backpropagate
        loss.backward()
        opt.step()

        # Project within L2 ball
        max_norm = hparams.clamp_norm_factor * target_init.norm()
        if delta.norm() > max_norm:
            with torch.no_grad():
                delta[...] = delta * max_norm / delta.norm()

    target = target_init + delta.to(device=target_init.device, dtype=target_init.dtype)
    print(
        f"Init norm {target_init.norm()} | Delta norm {delta.norm()} | Target norm {target.norm()}"
    )

    return target


def get_module_input_output_at_words(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    layer: int,
    context_templates: List[str],
    words: List[str],
    module_template: str,
    fact_token_strategy: str,
    track=None,
) -> Tuple[torch.Tensor]:
    """
    Retrieves detached representations for a word at the input and
    output of a particular layer module.
    """

    word_repr_args = dict(
        model=model,
        tok=tok,
        layer=layer,
        module_template=module_template,
    )
    if "subject_" in fact_token_strategy and fact_token_strategy.index("subject_") == 0:
        context_info = dict(
            context_templates=context_templates,
            words=words,
        )
        subtoken = fact_token_strategy[len("subject_") :]
        if track == 'out' or track == 'in':
            return repr_tools.get_reprs_at_word_tokens(
                track=track, subtoken=subtoken, **context_info, **word_repr_args
            )
        l_input, l_output = repr_tools.get_reprs_at_word_tokens(
            track="both", subtoken=subtoken, **context_info, **word_repr_args
        )
    elif fact_token_strategy == "last":
        raise Exception("This is definitely bugged, fix it.")
        context_info = dict(
            contexts=[
                tmp[i].format(words[i]) for i, tmp in enumerate(context_templates)
            ],
            idxs=[000000],
        )
        if track == 'out' or track == 'in':
            return repr_tools.get_reprs_at_word_tokens(
                track=track, subtoken=subtoken, **context_info, **word_repr_args
            )
        l_input, l_output = repr_tools.get_reprs_at_idxs(
            track="both", **context_info, **word_repr_args
        )
    else:
        raise ValueError(f"fact_token={fact_token_strategy} not recognized")

    return l_input.detach(), l_output.detach()


def find_fact_lookup_idx(
    prompt: str,
    subject: str,
    tok: AutoTokenizer,
    fact_token_strategy: str,
    verbose=True,
) -> int:
    """
    Computes hypothesized fact lookup index given a sentence and subject.
    """

    ret = None
    if fact_token_strategy == "last":
        ret = -1
    elif (
        "subject_" in fact_token_strategy and fact_token_strategy.index("subject_") == 0
    ):
        ret = repr_tools.get_words_idxs_in_templates(
            tok=tok,
            context_templates=[prompt],
            words=[subject],
            subtoken=fact_token_strategy[len("subject_") :],
        )[0][0]
    else:
        raise ValueError(f"fact_token={fact_token_strategy} not recognized")

    sentence = prompt.format(subject)
    if verbose:
        print(
            f"Lookup index found: {ret} | Sentence: {sentence} | Token:",
            tok.decode(tok(sentence)["input_ids"][ret]),
        )

    return ret
