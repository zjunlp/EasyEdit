from typing import Dict, List, Tuple

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from rome import repr_tools
from util import nethook
from random_word import RandomWords
import random
import copy
import time
from .hparams import EAMETHyperParams
import torch.nn.functional as F

# initial_sim_z = None

def compute_z(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    request: Dict,
    hparams: EAMETHyperParams,
    layer: int,
    context_templates: List[str],
    delta_list: List[torch.Tensor],
    combine_weights: List[float],
    request_id: int,
    layer_ks_norm: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Computes the value (right) vector for the rank-1 update.
    Runs a simple optimization procedure.
    """
    #noisy_num = max(noisy_num,len(request['subject'].split()))
    lm_w, ln_f = (
        nethook.get_parameter(model, f"{hparams.lm_head_module}.weight").T,
        nethook.get_module(model, hparams.ln_f_module),
    )

    # lm_w -> embedding.weight
    # lm_b -> potential embedding.bias
    # ln_f ->layer normalization at the final layer 
    try:
        lm_b = nethook.get_parameter(model, f"{hparams.lm_head_module}.bias")
        if lm_b == None:
            # same dtype and device with the precedent tensor
            lm_b = next(model.parameters()).new_zeros(model.config.vocab_size)
    except LookupError as _:
        lm_b = next(model.parameters()).new_zeros(model.config.vocab_size)

    if "gemma" not in str(model.config._name_or_path).lower():
        tok.add_bos_token = False
    else:
        tok.add_bos_token = True
    
    # Tokenize target into list of int token IDs
    # for the preadded str " "
    # Here, request["target_new"]["str"] should be " Negative". This is for the
    # correct word seperation. A sole space " " does not correspond to any token

    # tokenizer gives [[xxx]] for a single word

    ##############################################################################
    ### target_ids_test = tok(request["target_new"]["str"])["input_ids"]
    ### print(f"DEBUG INFO:target_ids_test: {target_ids_test}")
    ###
    ### The above codes give [xxx] instead of [[xxx]]
    ### return_tensors="pt" gives [[xxx]].
    ##############################################################################

    target_ids = tok(request["target_new"]["str"], return_tensors="pt").to("cuda")["input_ids"][0]

    print(f"DEBUG INFO:target_ids:{target_ids}")
    print(f"DEBUG INFO:tok('English'):{tok('English')}")
    print(f"DEBUG INFO:tok(' English'):{tok(' English')}")
    print(f"str(type(tok)):{str(type(tok))}")

    if "mistral" in str(model.config._name_or_path).lower() \
        or "qwen" in str(model.config._name_or_path).lower() \
        or "deepseek" in str(model.config._name_or_path).lower():
        opt_target_ids = target_ids
    elif "llama" in str(type(tok)) or \
        "llama-3.1" in str(model.config._name_or_path).lower() \
        or "gemma" in str(model.config._name_or_path).lower():
        opt_target_ids = target_ids[1:]
    else:
        opt_target_ids = target_ids
    
    print(f"opt_target_ids:{opt_target_ids}")
    tgt_str = request["target_new"]["str"]
    ### tokenizer.decode([]) gives nothing
    
    if "llama-3.1" in str(model.config._name_or_path).lower():
        tok_dcd = tok.decode(target_ids[:-1])
        print(f"tok_dcd:{tok_dcd}")
        bos_loc = tok_dcd.find("<|begin_of_text|>")
        bos_len = len("<|begin_of_text|>")
        tok_dcd = tok_dcd[bos_loc+bos_len:]
        rewriting_prompts, kl_prompts = [
            context.format(request["prompt"]) + tok_dcd
            for context_types in context_templates
            for context in context_types
        ], ["{} is a"]
    elif "gemma" in str(model.config._name_or_path).lower():
        rewriting_prompts, kl_prompts = [
            context.format(request["prompt"]) + tok.decode(opt_target_ids[:-1])
            for context_types in context_templates
            for context in context_types
        ], ["{} is a"]
    else:
        rewriting_prompts, kl_prompts = [
            context.format(request["prompt"]) + tok.decode(target_ids[:-1])
            for context_types in context_templates
            for context in context_types
        ], ["{} is a"]
    print(f"tok.decode(target_ids):{tok.decode(target_ids)}")
    print(f"rewriting_prompts:{rewriting_prompts}")

    all_prompts = rewriting_prompts + kl_prompts 
    subjects = [request['subject'] for i in range(len(rewriting_prompts))] + [request['subject']]
    all_filled_prompts = [prompt.format(subject) for prompt, subject in zip(all_prompts, subjects)]

    input_tok = tok(
        all_filled_prompts,
        return_tensors="pt",
        padding=True,
    ).to("cuda")

    rewriting_targets = torch.tensor(-100, device="cuda").repeat(
        len(rewriting_prompts), *input_tok["input_ids"].shape[1:]
    )
    for i in range(len(rewriting_prompts)):
        ex_len = input_tok["attention_mask"][i].sum()
        rewriting_targets[i, ex_len - len(opt_target_ids) : ex_len] = opt_target_ids

    lookup_idxs = [
        find_fact_lookup_idx(
            prompt, subject, tok, hparams.fact_token, verbose=False, model_name=str(model.config._name_or_path).lower()
        )
        for i, (prompt, subject) in enumerate(zip(all_prompts, subjects)) # not all filled prompts
    ]
    
    # Finalize rewrite and loss layers
    loss_layer = max(hparams.v_loss_layer, layer)
    delta = torch.zeros((model.config.hidden_size, ), device="cuda", requires_grad=True)
    target_init, kl_distr_init, target_constrain = None, None, None

    ### nonlocal helps to modify parameter from outer function
    def edit_output_fn(cur_out, cur_layer):
        nonlocal target_init, target_constrain, delta
        if cur_layer == hparams.layer_module_tmp.format(layer):
            if target_init is None:
                print("Recording initial value of v*")
                target_init = cur_out[0][0, lookup_idxs[0]].detach().clone()
                print(f"DEBUG INFO:layer_ks_norm:{layer_ks_norm}")

                if hparams.delta_init == "target_init":
                    if torch.norm(target_init) < 100:
                        ### not necessary for llama2 7b
                        init_data_copy = target_init.data.clone().detach()
                        init_data_copy = init_data_copy/torch.norm(init_data_copy)
                        init_data_copy = init_data_copy * layer_ks_norm
                        ### not necessary for llama2 7b

                        delta.data.copy_(init_data_copy)
                        print(f"SETTINGS INFO:Initializing delta with target_init")
                    else:
                        print(f"SETTINGS INFO:Initializing delta with zeros because of too large norm of target_init")
                elif hparams.delta_init == "zero":
                    print(f"SETTINGS INFO:Initializing delta with zeros")
                else:
                    raise ValueError(f"delta_init={hparams.delta_init} not recognized")
                    
            # Add intervened delta
            for i, idx in enumerate(lookup_idxs):
                cur_out[0][i, idx, :] += delta

        return cur_out

    # Optimizer
    opt = torch.optim.Adam([delta], lr=hparams.v_lr)
    nethook.set_requires_grad(False, model)

    # Execute optimization
    opt_steps = hparams.v_num_grad_steps
    scaled_max_norm = hparams.clamp_ks_factor * layer_ks_norm

    for it in range(opt_steps):
        opt.zero_grad()

        # Forward propagation
        with nethook.TraceDict(
            module=model,
            layers=[
                hparams.layer_module_tmp.format(layer),
                hparams.layer_module_tmp.format(loss_layer),
            ],
            retain_input=False,
            retain_output=True,
            edit_output=edit_output_fn,
        ) as tr:
            logits = model(**input_tok).logits

            # Compute distribution for KL divergence
            kl_logits = torch.stack(
                [
                    logits[-len(kl_prompts)+i, idx, :]
                    for i, idx in enumerate(lookup_idxs[-len(kl_prompts):])
                ],
                dim=0,
            )

            kl_log_probs = torch.nn.functional.log_softmax(kl_logits, dim=1)
            if kl_distr_init is None:
                kl_distr_init = kl_log_probs.detach().clone()

        # Compute loss on rewriting targets
        full_repr = tr[hparams.layer_module_tmp.format(loss_layer)].output[0][
            : len(rewriting_prompts)
        ]

        after_mult = ln_f(full_repr) @ lm_w + lm_b
        log_probs = torch.log_softmax(after_mult, dim=2)

        loss = torch.gather(
            log_probs,
            2,
            torch.where(rewriting_targets != -100, rewriting_targets, 0).unsqueeze(2),
        ).squeeze(2) # use gather to pick desired token from vocabulary
        mask = (rewriting_targets != -100).float()

        # Aggregate total losses
        
        sample_cs_collision = combine_weights[request_id]
        if len(delta_list) == 0:
            collision_loss = 0
            mse_loss = 0
        else:
            avoid_z_ind = [i for i in range(len(delta_list))] 
            if it == 1:
                start_time = time.time()
            collision_loss, mse_loss = keep_cs_structure(delta_list, 
                                                         avoid_z_ind, 
                                                         sample_cs_collision, 
                                                         delta, 
                                                         tau=hparams.tau)
            collision_loss = hparams.cs_factor * collision_loss
            mse_loss = hparams.mse_factor * mse_loss
            if it == 1:
                end_time = time.time()
                print(f"time cost when compute cs structure:{end_time - start_time}")

        nll_loss_each = -(loss * mask).sum(1) / opt_target_ids.size(0)
        nll_loss = nll_loss_each.mean()

        kl_loss = hparams.kl_factor * torch.nn.functional.kl_div(
            kl_distr_init, kl_log_probs, log_target=True, reduction="batchmean"
        )

        if hparams.weight_decay_method == "norm":
            weight_decay = hparams.v_weight_decay * (
                torch.norm(delta) / torch.norm(target_init)
            )
        elif hparams.weight_decay_method == "fix_ks":
            weight_decay = hparams.norm_factor * (torch.norm(delta) - scaled_max_norm) ** 2
        else:
            raise ValueError(f"weight_decay_method={hparams.weight_decay_method} not recognized")

        loss = nll_loss + kl_loss + weight_decay + collision_loss + mse_loss

        if loss < 1e-2 or it==0:
            if isinstance(collision_loss, int):
                print(
                    f"loss {np.round(loss.item(), 3)} = {np.round(nll_loss.item(), 3)}(nll) + {np.round(kl_loss.item(), 3)}(kl) + {np.round(weight_decay.item(), 3)}(weight_decay) + {np.round(collision_loss, 3)}(collision) + {np.round(mse_loss, 3)}(mse) "
                    f"avg prob of [{tgt_str}] "
                    f"{torch.exp(-nll_loss_each).mean().item()}"
                )
            else:
                print(
                    f"loss {np.round(loss.item(), 3)} = {np.round(nll_loss.item(), 3)}(nll) + {np.round(kl_loss.item(), 3)}(kl) + {np.round(weight_decay.item(), 3)}(weight_decay) + {np.round(collision_loss.item(), 3)}(collision) + {np.round(mse_loss.item(), 3)}(mse) "
                    f"avg prob of [{tgt_str}] "
                    f"{torch.exp(-nll_loss_each).mean().item()}"
                )
            if len(delta_list) != 0:
                _, _ = keep_cs_structure(delta_list, 
                                         avoid_z_ind, 
                                         sample_cs_collision, 
                                         delta, 
                                         tau=hparams.tau, 
                                         islog=True)
            if it != 0:
                break
        if it == opt_steps - 1:
            if isinstance(collision_loss, int):
                print(
                    f"loss {np.round(loss.item(), 3)} = {np.round(nll_loss.item(), 3)}(nll) + {np.round(kl_loss.item(), 3)}(kl) + {np.round(weight_decay.item(), 3)}(weight_decay) + {np.round(collision_loss, 3)}(collision) + {np.round(mse_loss, 3)}(mse) "
                    f"avg prob of [{tgt_str}] "
                    f"{torch.exp(-nll_loss_each).mean().item()}"
                )
            else:
                print(
                    f"loss {np.round(loss.item(), 3)} = {np.round(nll_loss.item(), 3)}(nll) + {np.round(kl_loss.item(), 3)}(kl) + {np.round(weight_decay.item(), 3)}(weight_decay) + {np.round(collision_loss.item(), 3)}(collision) + {np.round(mse_loss.item(), 3)}(mse) "
                    f"avg prob of [{tgt_str}] "
                    f"{torch.exp(-nll_loss_each).mean().item()}"
                )
            if len(delta_list) != 0:
                _, _ = keep_cs_structure(delta_list, 
                                         avoid_z_ind, 
                                         sample_cs_collision, 
                                         delta, 
                                         tau=hparams.tau, 
                                         islog=True)
            break

        # Backpropagate
        loss.backward()
        opt.step()

        # Project within L2 ball
        if hparams.weight_decay_method == "fix_ks":
            max_norm = scaled_max_norm
            if it == 0:
                print(f"SETTINGS INFO:regularize norm using fix_ks")
        elif hparams.weight_decay_method == "norm":
            max_norm = hparams.clamp_norm_factor * target_init.norm()
            if it == 0: 
                print(f"SETTINGS INFO:regularize norm using norm")
        else:
            raise ValueError(f"weight_decay_method={hparams.weight_decay_method} not recognized")

        if delta.norm() > max_norm:
            with torch.no_grad():
                delta[...] = delta * max_norm / delta.norm()
 
    target = target_init + delta    
    print(
        f"Init norm {target_init.norm()} | Delta norm {delta.norm()} | Target norm {target.norm()}"
    )
    print(f" ")

    return target.detach().cpu(), delta.detach().cpu()

def noisy_trigger(
        trigger: str,
        num: int,
        tok: AutoTokenizer
) -> List[torch.Tensor]:
    gen = RandomWords()
    trigers = tok([trigger], return_tensors="pt",
        padding=False)['input_ids'][0].tolist()
    noisy_list = []
    for i in range(len(trigers)):
        if i == len(trigers) - 1:
            noisy_list.append(tok.decode(torch.tensor(trigers[:-1])))
        else:
            noisy_list.append(tok.decode(torch.tensor(trigers[0:i] + trigers[i + 1:])))
    return noisy_list


def get_module_input_output_at_words(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    layer: int,
    context_templates: List[str],
    words: List[str],
    module_template: str,
    fact_token_strategy: str,
    minus = None,
    change_padding = False
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
        change_padding=change_padding
    )
    if "subject_" in fact_token_strategy and fact_token_strategy.index("subject_") == 0:
        context_info = dict(
            context_templates=context_templates,
            words=words,
        )
        subtoken = fact_token_strategy[len("subject_") :]
        l_input, l_output = repr_tools.get_reprs_at_word_tokens(
            track="both", subtoken=subtoken, minus=minus, **context_info, **word_repr_args
        )
    elif fact_token_strategy == "last":
        raise Exception("This is definitely bugged, fix it.")
        context_info = dict(
            contexts=[
                tmp[i].format(words[i]) for i, tmp in enumerate(context_templates)
            ],
            idxs=[000000],
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
    model_name: str = None
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
            model_name=model_name
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

def topk_cs(similarities: torch.Tensor, k: int) -> List[int]:
    k = min(k, similarities.size(0))
    _, indices = torch.topk(similarities, k)
    return indices.tolist()

def keep_cs_structure(delta_list: List[torch.Tensor], 
                      avoid_z_ind: List[int], 
                      k_similarity: torch.Tensor,
                      delta: torch.Tensor = None,
                      tau: float = 0.07,      # temperature for softmax
                      eps: float = 1e-8,       # avoid log(0) 
                      islog: bool = False
                      ) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    KL-divergence based structure-preserving loss between cosine similarities.
    """
    r_new = delta            # shape: [d]
    r_new = F.normalize(r_new, dim=0)      # ensure normalized

    # Gather r_i vectors for selected indices
    r_candidates = torch.stack([delta_list[i] for i in avoid_z_ind], dim=0)  # [n, d]
    r_candidates = F.normalize(r_candidates, dim=1)

    # 1. Predicted similarity (q): softmax of cos(r_new, r_i)
    sim_r = torch.matmul(r_candidates, r_new) # [n]
    q = F.softmax(sim_r/tau, dim=0)                                         # [n]

    # 2. Target similarity (p): softmax of provided k-similarity
    sim_k = k_similarity[avoid_z_ind]
    p = F.softmax(sim_k/tau, dim=0)                                       # [n]

    # 3. KL divergence: KL(p || q)
    kl = torch.sum(p * torch.log((p + eps) / (q + eps)))

    # 4. MSE: MSE(p, sim_k)
    topk_idx = topk_cs(sim_k, 50)
    sim_k_topk = sim_k[topk_idx].detach()            # [k]
    sim_r_topk = sim_r[topk_idx]                     # [k]

    # Weight with softmax over sim_k_topk
    p_topk = F.softmax(sim_k_topk / tau, dim=0)     # [k]

    # Weighted MSE
    mse = torch.sum(p_topk * (sim_k_topk - sim_r_topk) ** 2)

    if islog: 
        # print(f"DEBUG INFO:sim_k:{sim_k}")
        # print(f"DEBUG INFO:sim_r:{sim_r}")
        print(f"DEBUG INFO:last sim_k :{sim_k[topk_idx[-1]]}")
        print(f"DEBUG INFO:tau :{tau}") 
        
    return kl, mse
