import unicodedata
from typing import List, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .logit_lens import LogitLens


def generate_interactive(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    top_k: int = 5,
    max_out_len: int = 200,
    compare_against: Optional[AutoModelForCausalLM] = None,
    use_logit_lens: bool = False,
    layer_module_tmp: str = "transformer.h.{}",
    ln_f_module: str = "transformer.ln_f",
    lm_head_module: str = "lm_head",
):
    """
    Puts generation in a loop. Allows users to repeatedly provide inputs
    with which text is generated.
    """

    if use_logit_lens:
        llens_gen = LogitLens(
            model,
            tok,
            layer_module_tmp,
            ln_f_module,
            lm_head_module,
            disabled=not use_logit_lens,
        )
        if compare_against:
            llens_vanilla = LogitLens(
                compare_against,
                tok,
                layer_module_tmp,
                ln_f_module,
                lm_head_module,
                disabled=not use_logit_lens,
            )

    while True:
        prompt = input("Enter a prompt: ").strip(" \r\t\n")

        print(
            f"Argument Model: "
            f"{generate_fast(model, tok, [prompt], n_gen_per_prompt=1, top_k=top_k, max_out_len=max_out_len)}"
        )
        if compare_against:
            print(
                f"Baseline Model: "
                f"{generate_fast(compare_against, tok, [prompt], n_gen_per_prompt=1, top_k=top_k, max_out_len=max_out_len)}"
            )

        if use_logit_lens:
            inp_prompt = tok([prompt], padding=True, return_tensors="pt").to(
                next(model.parameters()).device
            )

            with llens_gen:
                model(**inp_prompt)
            print("\n--- Argument Model Logit Lens ---")
            llens_gen.pprint()

            if compare_against:
                with llens_vanilla:
                    compare_against(**inp_prompt)
                print("--- Baseline Model Logit Lens ---")
                llens_vanilla.pprint()

        print()


def generate_fast(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    prompts: List[str],
    n_gen_per_prompt: int = 1,
    top_k: int = 5,
    max_out_len: int = 200,
    vanilla_generation=False,
):
    """
    Fast, parallelized auto-regressive text generation with top-k sampling.
    Our custom implementation.
    """

    # Unroll prompts and tokenize
    inp = [prompt for prompt in prompts for _ in range(n_gen_per_prompt)]
    inp_tok = tok(inp, padding=True, return_tensors="pt").to(
        next(model.parameters()).device
    )
    input_ids, attention_mask = inp_tok["input_ids"], inp_tok["attention_mask"]
    if vanilla_generation:
        gen_txt = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_out_len
        )
        txt = [tok.decode(x, skip_special_tokens=True) for x in gen_txt.detach().cpu().numpy().tolist()]
        txt = [
            unicodedata.normalize("NFKD", x)
            .replace("\n\n", " ")
            .replace("<|endoftext|>", "")
            for x in txt
        ]
        return txt
    batch_size = input_ids.size(0)

    # Setup storage of fast generation with attention caches.
    # `cur_context` is used to define the range of inputs that are not yet
    # stored in `past_key_values`. At each step, we are generating the
    # next token for the index at `cur_context.stop + 1`.
    past_key_values, cur_context = None, slice(0, attention_mask.sum(1).min().item())

    with torch.no_grad():
        while input_ids.size(1) < max_out_len:  # while not exceeding max output length
            model_out = model(
                input_ids=input_ids[:, cur_context],
                attention_mask=None if 'llama'or'baichuan' in model.name_or_path.lower() else attention_mask[:, cur_context],
                past_key_values=past_key_values,
                use_cache=True,
            )
            if type(model_out) is torch.Tensor:
                logits = outputs
            else:
                logits = outputs.logits
            past_key_values = model_out.logits, model_out.past_key_values
            softmax_out = torch.nn.functional.softmax(logits[:, -1, :], dim=1)

            # Top-k sampling
            tk = torch.topk(softmax_out, top_k, dim=1).indices
            softmax_out_top_k = torch.gather(softmax_out, 1, tk)
            softmax_out_top_k = softmax_out_top_k / softmax_out_top_k.sum(1)[:, None]
            new_tok_indices = torch.multinomial(softmax_out_top_k, 1)
            new_toks = torch.gather(tk, 1, new_tok_indices)

            # If we're currently generating the continuation for the last token in `input_ids`,
            # create a new index so we can insert the new token
            if cur_context.stop == input_ids.size(1):
                attention_mask = torch.cat(
                    [attention_mask, attention_mask.new_zeros(batch_size, 1)], dim=1
                )
                input_ids = torch.cat(
                    [
                        input_ids,
                        input_ids.new_ones(batch_size, 1) * tok.pad_token_id,
                    ],
                    dim=1,
                )

            last_non_masked = attention_mask.sum(1) - 1
            for i in range(batch_size):
                new_idx = last_non_masked[i] + 1
                if last_non_masked[i].item() + 1 != cur_context.stop:
                    continue

                # Stop generating if we've already maxed out for this prompt
                if new_idx < max_out_len:
                    input_ids[i][new_idx] = new_toks[i]
                    attention_mask[i][new_idx] = 1

            cur_context = slice(cur_context.stop, cur_context.stop + 1)
    txt = [tok.decode(x, skip_special_tokens=True) for x in input_ids.detach().cpu().numpy().tolist()]
    txt = [
        unicodedata.normalize("NFKD", x)
        .replace("\n\n", " ")
        .replace("<|endoftext|>", "")
        for x in txt
    ]

    return txt
