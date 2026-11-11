import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from ...util import nethook

from .Revised_AlphaEdit_hparams import Revised_AlphaEditHyperParams


def compute_z(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    request: dict,
    hparams: Revised_AlphaEditHyperParams,
    layer: int,
    context_templates: list[str],
    return_statistics: bool = False,
):
    """
    Computes the value (right) vector for the rank-1 update.
    Runs a simple optimization procedure.
    """

    # Get model parameters
    lm_w, ln_f = (
        nethook.get_module(model, f"{hparams.lm_head_module}").weight.T,
        nethook.get_module(model, hparams.ln_f_module),
    )
    input_device = model.get_input_embeddings().weight.device
    rewrite_device = next(
        nethook.get_module(model, hparams.layer_module_tmp.format(layer)).parameters()
    ).device
    try:
        lm_b = nethook.get_parameter(model, f"{hparams.lm_head_module}.bias")
    except LookupError as _:
        lm_b = next(model.parameters()).new_zeros(model.config.vocab_size)

    print("Computing right vector (v)")
    
    # Compile list of rewriting and KL x/y pairs
    rewriting_prompts = []
    for context_types in context_templates:
        for context in context_types:
            prompt = context.replace("{prompt}", request["prompt"])
            rewriting_prompts.append(prompt)
    kl_prompts = [f"{request['subject']} is a"]
    
    all_prompts = []
    for rewriting_prompt in rewriting_prompts:
        all_prompts.append(f"{rewriting_prompt} {request['target_new']}")
    
    all_prompts.extend(kl_prompts)

    tok.padding_side = "right"
    input_tok = tok(
        all_prompts,
        return_tensors="pt",
        padding=True,
        add_special_tokens=True,
    ).to(input_device)

    # Compute rewriting targets
    rewriting_targets = torch.tensor(-100, device=input_device).repeat(
        len(rewriting_prompts), input_tok["input_ids"].shape[1]
    )
    for i in range(len(rewriting_prompts)):
        start_char = all_prompts[i].find(request["target_new"])
        end_char = start_char + len(request["target_new"]) - 1
        start_tok = input_tok.char_to_token(i, start_char)
        end_tok = input_tok.char_to_token(i, end_char)
        rewriting_targets[i, start_tok - 1 : end_tok] = input_tok["input_ids"][i, start_tok : end_tok + 1]
    
    # Compute indices of the tokens where the fact is looked up
    lookup_idxs = []
    if hparams.fact_token == "subject_first":
        for i in range(len(all_prompts)):
            start_char = all_prompts[i].find(request["subject"])
            start_tok = input_tok.char_to_token(i, start_char)
            lookup_idxs.append(start_tok)
    elif hparams.fact_token == "subject_last":
        for i in range(len(all_prompts)):
            start_char = all_prompts[i].find(request["subject"])
            end_char = start_char + len(request["subject"]) - 1
            end_tok = input_tok.char_to_token(i, end_char)
            lookup_idxs.append(end_tok)
    print(f"lookup_idxs: {lookup_idxs}")
    for i, lookup_idx in enumerate(lookup_idxs):
        print(f"Prompt {i}: {tok.decode(input_tok['input_ids'][i][:lookup_idx + 1])}")

    # Finalize rewrite and loss layers
    loss_layer = max(hparams.v_loss_layer, layer)
    print(f"Rewrite layer is {layer}")
    print(f"Tying optimization objective to layer: {loss_layer}")
    print(f"Grad steps is: {hparams.v_num_grad_steps}")

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

    i_idx = range(len(lookup_idxs))
    # Inserts new "delta" variable at the appropriate part of the computation
    def edit_output_fn(cur_out, cur_layer):
        nonlocal target_init

        if cur_layer == hparams.layer_module_tmp.format(layer):
            # Store initial value of the vector of interest
            if target_init is None:
                print("Recording initial value of v*")
                # Initial value is recorded for the clean sentence
                if isinstance(cur_out, torch.Tensor):
                    # Tested: meta-llama/Meta-Llama-3-8B
                    target_init = cur_out[0, lookup_idxs[0]].detach().clone()
                else:
                    target_init = cur_out[0][0, lookup_idxs[0]].detach().clone()

            # Add intervened delta
            if isinstance(cur_out, torch.Tensor):
                cur_out[i_idx, lookup_idxs, :] += delta
            elif len(lookup_idxs) != len(cur_out[0]):
                cur_out[0][lookup_idxs, i_idx, :] += delta
            else:
                cur_out[0][i_idx, lookup_idxs, :] += delta

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
            num_kl = len(kl_prompts)
            batch_idxs = list(range(-num_kl, 0))
            seq_idxs = lookup_idxs[-num_kl:]
            kl_logits = logits[batch_idxs, seq_idxs, :]
            kl_log_probs = torch.nn.functional.log_softmax(kl_logits, dim=1)
            if kl_distr_init is None:
                kl_distr_init = kl_log_probs.detach().clone()

        # Compute loss on rewriting targets
        output = tr[hparams.layer_module_tmp.format(loss_layer)].output
        if isinstance(output, tuple):
            output = output[0]
        expected_batch_size = len(all_prompts)
        expected_sequence_length = rewriting_targets.shape[1]
        if output.ndim != 3:
            raise RuntimeError(
                "Expected traced loss-layer output to have shape "
                f"[batch, sequence, hidden] or [sequence, batch, hidden], got {tuple(output.shape)}."
            )
        if output.shape[:2] == (expected_sequence_length, expected_batch_size):
            output = torch.transpose(output, 0, 1)
        elif output.shape[:2] != (expected_batch_size, expected_sequence_length):
            raise RuntimeError(
                "Traced loss-layer output does not match the tokenized prompt dimensions: "
                f"output shape={tuple(output.shape)}, expected batch={expected_batch_size}, "
                f"expected sequence={expected_sequence_length}."
            )
        full_repr = output[:len(rewriting_prompts)]

        log_probs = torch.log_softmax(ln_f(full_repr) @ lm_w.to(full_repr.device) + lm_b.to(full_repr.device), dim=2) # shape [batch, seq_len, vocab_size]
        target_ids = rewriting_targets.to(log_probs.device)
        mask = (target_ids != -100).float()
        loss = torch.gather(
            log_probs,
            2,
            torch.where(target_ids != -100, target_ids, 0).unsqueeze(2),
        ).squeeze(2) # shape [batch, seq_len]

        # Aggregate total losses
        nll_loss_each = -(loss * mask).sum(1) / mask.sum(1)
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

    target = target_init + delta
    print(
        f"Init norm {target_init.norm()} | Delta norm {delta.norm()} | Target norm {target.norm()}"
    )

    response = {
        "cur_z": target.detach(),
        "target_init": target_init.detach(),
    }

    if return_statistics:
        final_statistics = {
            "delta_vector_loss": float(loss.detach().cpu()),
            "nll_loss": float(nll_loss.detach().cpu()),
            "kl_loss": float(kl_loss.detach().cpu()),
            "weight_decay": float(weight_decay.detach().cpu()),
            "optimization_steps": it + 1,
            "delta_vector_norm": float(delta.norm().detach().cpu()),
            "target_init_norm": float(target_init.norm().detach().cpu()),
            "target_norm": float(target.norm().detach().cpu()),
        }
        response["target_statistics"] = final_statistics

    return response