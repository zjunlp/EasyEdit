from typing import Dict, List, Tuple
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from .unke_are_hparams import UnkeAREHyperParams
from ...util import nethook
import nltk


def compute_z(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    data: Dict,
    layer: int,
    hparams: UnkeAREHyperParams,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Computes the value (right) vector for the rank-1 update.
    Runs a simple optimization procedure.
    """

    # Get model parameters (bs:seq:h_dim) -> (bs:seq:vocab_size)
    lm_w, ln_f = (
        nethook.get_parameter(model, f"{hparams.lm_head_module}.weight").T,
        nethook.get_module(model, hparams.ln_f_module),
    )
    try:
        lm_b = nethook.get_parameter(model, f"{hparams.lm_head_module}.bias")
    except LookupError as _:
        lm_b = next(model.parameters()).new_zeros(model.config.vocab_size)

    # Tokenize target into list of int token IDs
    target_ids = tok(data["answer"], return_tensors="pt").to(f"cuda:{hparams.device}")[
        "input_ids"
    ][0]  
    
    if target_ids[0] == tok.bos_token_id or target_ids[0] == tok.unk_token_id:
        target_ids = target_ids[1:]
    
    input_tok = tok(
        [data["question"]],  
        return_tensors="pt",
        padding=True,
    ).to(f"cuda:{hparams.device}")

    cur_input_ids = input_tok['input_ids'] 
    all_delta = []
    all_target = []
    all_idxs = []

    start = 0
    while start < len(target_ids):
        end = start + hparams.window_size
        if end > len(target_ids):
            end = len(target_ids)
        current_target_ids = target_ids[start:end]
        if start > 0:
            input_ids = torch.cat([cur_input_ids, torch.unsqueeze(current_target_ids[hparams.overlap:-1], dim=0)], dim=1)
            cur_input_ids = torch.cat([cur_input_ids, torch.unsqueeze(current_target_ids[hparams.overlap:], dim=0)], dim=1)
        else:
            input_ids = torch.cat([cur_input_ids, torch.unsqueeze(current_target_ids[:-1], dim=0)], dim=1)
            cur_input_ids = torch.cat([cur_input_ids, torch.unsqueeze(current_target_ids, dim=0)], dim=1)
        
        start += hparams.window_size - hparams.overlap
        
        rewriting_targets = torch.tensor(-100, device=f"cuda:{hparams.device}").repeat(
            1, len(input_ids[0])
        )
   
        ex_len = len(input_ids[0])
    
        rewriting_targets[0, ex_len - len(current_target_ids) : ex_len] = current_target_ids

        lookup_idxs = [ex_len - len(current_target_ids)]
    
        loss_layer = max(hparams.v_loss_layer, layer)
    
        if hasattr(model.config, 'n_embd'):
            delta = torch.zeros((model.config.n_embd,), requires_grad=True, device=f"cuda:{hparams.device}")
        elif hasattr(model.config, 'hidden_size'):
            delta = torch.zeros((model.config.hidden_size,), requires_grad=True, device=f"cuda:{hparams.device}")
        else:
            raise NotImplementedError
        target_init = None
    
        def edit_output_fn(cur_out, cur_layer):
            nonlocal target_init  

            if cur_layer == hparams.layer_module_tmp.format(layer):
                
                if target_init is None:
                
                    target_init = cur_out[0][0, lookup_idxs[0]].detach().clone()

                for idxs_pre, delta_pre in all_delta:
                    for i, idx in enumerate(idxs_pre):
                        if len(idxs_pre)!=len(cur_out[0]):
                            cur_out[0][idx, i, :] += delta_pre
                        else:
                            cur_out[0][i, idx, :] += delta_pre
                for i, idx in enumerate(lookup_idxs):
                    
                    if len(lookup_idxs)!=len(cur_out[0]):
                        cur_out[0][idx, i, :] += delta
                    else:
                        cur_out[0][i, idx, :] += delta

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
                logits = model(input_ids).logits
                
            # Compute loss on rewriting targets
            output = tr[hparams.layer_module_tmp.format(loss_layer)].output[0]  
            if output.shape[1] != rewriting_targets.shape[1]:
                output = torch.transpose(output, 0, 1)
            full_repr = output

            log_probs = torch.log_softmax(ln_f(full_repr) @ lm_w.to(full_repr.device) + lm_b.to(full_repr.device), dim=2)
            loss = torch.gather(
                log_probs,
                2,
                torch.where(rewriting_targets != -100, rewriting_targets, 0).unsqueeze(2).to(log_probs.device),
            ).squeeze(2)
            mask = (rewriting_targets != -100).float()

            # Aggregate total losses
            nll_loss_each = -(loss * mask.to(loss.device)).sum(1) / current_target_ids.size(0)
            nll_loss = nll_loss_each.mean()
            
            weight_decay = hparams.v_weight_decay * (
                torch.norm(delta) / torch.norm(target_init) ** 2
            )
            # weight_decay = hparams.v_weight_decay * torch.norm(delta) ** 2
            loss = nll_loss + weight_decay.to(nll_loss.device)
            
            if loss < 5e-3:
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
                    
        target = target_init + delta  
        all_delta.append((lookup_idxs, delta.clone()))
        all_target.append(target)
        all_idxs.append(lookup_idxs[0])

    return all_idxs, all_target 