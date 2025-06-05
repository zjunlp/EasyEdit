
from typing import Dict, List, Tuple  # 추가: Literal type
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from ..rome import repr_tools
from ...util import nethook

from .core_hparams import COREHyperParams

from ...util.generate import generate_fast
from itertools import combinations



def compute_z(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    request: Dict,
    hparams: COREHyperParams,
    layer: int,
    context_templates: List[str],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Computes the value (right) vector for the rank-1 update.
    Runs a simple optimization procedure.
    """

    # New Context_templates 
    if hparams.context is not None : 
        context_templates = get_context_templates(model, tok, request, context_type=hparams.context , n_gen=hparams.ctx_num, max_length= hparams.ctx_len, ctx_top_k= hparams.ctx_top_k)

    print(context_templates)

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

    # Tokenize target into list of int token IDs
    target_ids = tok.encode(request["target_new"], return_tensors="pt", add_special_tokens=False).to(f"cuda:{hparams.device}")[0]

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
    ).to(f"cuda:{hparams.device}")

    # Compute rewriting targets
    rewriting_targets = torch.tensor(-100, device=f"cuda:{hparams.device}").repeat(
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
        delta = torch.zeros((model.config.n_embd,), requires_grad=True, device=f"cuda:{hparams.device}")
    elif hasattr(model.config, 'hidden_size'):
        delta = torch.zeros((model.config.hidden_size,), requires_grad=True, device=f"cuda:{hparams.device}")
    else:
        raise NotImplementedError
    target_init, kl_distr_init = None, None

    # Inserts new "delta" variable at the appropriate part of the computation
    def edit_output_fn(cur_out, cur_layer):
        nonlocal target_init

        if cur_layer == hparams.layer_module_tmp.format(layer):
            # Store initial value of the vector of interest
            if target_init is None:
                print("Recording initial value of v*")
                # Initial value is recorded for the clean sentence
                target_init = cur_out[0][0, lookup_idxs[0]].detach().clone()

            # Add intervened delta
            for i, idx in enumerate(lookup_idxs):

                if len(lookup_idxs)!=len(cur_out[0]):
                    cur_out[0][idx, i, :] += delta
                else:
                    cur_out[0][i, idx, :] += delta

        return cur_out

    # Optimizer
    opt = torch.optim.Adam([delta], lr=hparams.v_lr)
    nethook.set_requires_grad(False, model)


    # For Prefix Regularization term 
    num_layers = model.config.num_hidden_layers
    monitor_layers = []
    range_ = 1 + hparams.layer_range
    for extra_ly in range(layer+1, layer+range_):
        if extra_ly < num_layers:  
            monitor_layers.append(extra_ly)
    layers_to_trace = [
        hparams.layer_module_tmp.format(loss_layer),
        hparams.layer_module_tmp.format(layer),
    ]
    for ml in monitor_layers:
        layers_to_trace.append(hparams.layer_module_tmp.format(ml))


    # Execute optimization
    for it in range(hparams.v_num_grad_steps):

        opt.zero_grad()

        # Forward propagation
        with nethook.TraceDict(
            module=model,
            layers=layers_to_trace,
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

        output=tr[hparams.layer_module_tmp.format(loss_layer)].output[0]
        if output.shape[1]!=rewriting_targets.shape[1]:
            output=torch.transpose(output, 0, 1)
        full_repr = output[:len(rewriting_prompts)]

        log_probs = torch.log_softmax(ln_f(full_repr) @ lm_w.to(full_repr.device) + lm_b.to(full_repr.device), dim=2)
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


        # For Prefix Regularization term 
        prefix_consistency_loss = torch.tensor(0.0, device=nll_loss.device)
        prefix_reg_lambda = hparams.reg_lambda
        if prefix_reg_lambda > 0:
            all_layer_outputs = []
            for mon_ly in monitor_layers:
                layer_name = hparams.layer_module_tmp.format(mon_ly)
                monitored_out = tr[layer_name].output[0]
                all_layer_outputs.append(monitored_out)
            
            all_layer_outputs = torch.stack(all_layer_outputs, dim=0).to(nll_loss.device)
            num_prompts = len(rewriting_prompts) 
            batch_indices = torch.arange(num_prompts, device=nll_loss.device)  # 0 ~ N-1
            prompt_indices = torch.LongTensor(lookup_idxs[:num_prompts]).to(nll_loss.device) 
            prompt_hiddens = all_layer_outputs[:, batch_indices, prompt_indices, :]
            L, N, D = prompt_hiddens.shape
            
            norms = prompt_hiddens.pow(2).sum(dim=-1)  
            sum_vectors = prompt_hiddens.sum(dim=1)           
            sum_pairwise_sq = N * norms.sum(dim=1) - sum_vectors.pow(2).sum(dim=1)           
            sum_pairwise_sq /= D  # shape: [L]            
            prefix_consistency_loss = sum_pairwise_sq.sum()
            
            num_layers = len(monitor_layers)  
            prefix_consistency_loss = sum_pairwise_sq.sum() / num_layers
            prefix_consistency_loss *= prefix_reg_lambda


        loss = nll_loss + kl_loss.to(nll_loss.device) + weight_decay.to(nll_loss.device) + prefix_consistency_loss.to(nll_loss.device)  # 추가
        print(
                f"It {it}, loss {np.round(loss.item(), 3)} = {np.round(nll_loss.item(), 3)} + "
                f"{np.round(kl_loss.item(), 3)} + {np.round(weight_decay.item(), 3)} + "
                f"{np.round(prefix_consistency_loss.item(), 3)} | prefix_consistency_loss: {prefix_consistency_loss.item():.3f} | "
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
    return target


def get_context_templates(model, tok, request, context_type='all', n_gen=5, max_length=10, ctx_top_k: int = 5):
    """
    Unified function to generate context templates.

    Args:
        model: model to use
        tok: tokenizer
        request: request data dictionary
        context_type: type of context to generate ('all', 'target_true', 'target_new', 'subject', 'target_true_n_subject')
        n_gen: number of sentences to generate (default: 5)
        max_length: maximum sentence length (default: 10)
    """
    global CONTEXT_TEMPLATES_CACHE

    # 1) Added context_type validation
    valid_types = {'all', 'target_true', 'target_new', 'subject', 'target_true_n_subject'}
    if context_type not in valid_types:
        raise ValueError(f"Invalid context_type: {context_type}")

    print("Starting context template generation...")

    # 2) Prompt setup and calculation of n_gen_per_prompt (ensure at least 1)
    if context_type == 'all':
        ground_truth = request["ground_truth"]
        subject      = request["subject"]
        target_new   = request["target_new"]
        all_prompts  = [ground_truth, subject, target_new]
        n_gen_per_prompt = max(n_gen // len(all_prompts), 1)

    elif context_type == 'target_true_n_subject':
        ground_truth = request["ground_truth"]
        subject      = request["subject"]
        all_prompts  = [ground_truth, subject]
        n_gen_per_prompt = max(n_gen // len(all_prompts), 1)

    else:
        prompt_key = {
            'target_true': 'ground_truth',
            'target_new' : 'target_new',
            'subject'    : 'subject'
        }
        target = request[prompt_key[context_type]]
        all_prompts = [target]
        n_gen_per_prompt = n_gen

    print(f"Start prompts: {all_prompts}")

    # 3) Prevent variable shadowing: use underscore (_) for the second value
    gen_configs = [(max_length, n_gen)]

    # 4) Generate actual templates (removed duplicate calls)
    CONTEXT_TEMPLATES_CACHE = [["{}"]] + [
        [
            template.replace("{", " ").replace("}", " ") + ". {}"
            for template in generate_fast(
                model,
                tok,
                all_prompts,
                n_gen_per_prompt=n_gen_per_prompt,
                max_out_len=length,
                top_k=ctx_top_k
            )
        ]
        for length, _ in gen_configs
    ]

    print(f"Final context templates: {CONTEXT_TEMPLATES_CACHE}")
    return CONTEXT_TEMPLATES_CACHE





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