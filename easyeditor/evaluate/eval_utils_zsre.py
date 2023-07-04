"""
Contains evaluation utilities for pytorch-based rewriting methods.
To use, simply call `compute_rewrite_quality_zsre` with the
appropriate arguments, which returns a dictionary containing them.
"""

import typing
from itertools import chain

import numpy as np
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import AutoModelForCausalLM, AutoTokenizer


def compute_rewrite_quality_zsre(
    model,
    model_name,
    tok: AutoTokenizer,
    record: typing.Dict,
) -> typing.Dict:
    """
    Given a rewritten model, computes generalization and specificity metrics for
    the desired rewrite (passed in via the CounterFact dataset record). Returns a
    dictionary containing those metrics.

    :param model: Rewritten model
    :param tok: Tokenizer
    :param record: CounterFact dataset record
    :paran snips: ???
    :param vec: ???
    :return: Dictionary containing rewriting metrics
    """

    # First, unpack rewrite evaluation record.
    subject, target_new, target_true = (
        record["requested_rewrite"][x] for x in ["subject", "target_new", "target_true"]
    )
    rewrite_prompts = [record["requested_rewrite"]["prompt"].format(subject)]
    paraphrase_prompts = record["paraphrase_prompts"]
    neighborhood_prompts = record["neighborhood_prompts"]

    # Form a list of lists of prefixes to test.
    prob_prompts = [
        rewrite_prompts,
        paraphrase_prompts,
    ]
    # Flatten all the evaluated prefixes into one list.
    if 't5' in model_name.lower():
        stuff_probs = test_seq2seq_batch_prediction_acc(model, tok,
                                                        list(chain(*prob_prompts)),
                                                        target_new["str"])
        neighborhood_correct = test_seq2seq_batch_prediction_acc(model, tok,
                                                                 [neighborhood_prompts['prompt']],
                                                                 neighborhood_prompts["target"],
                                                                 neighborhood=True)
        cutoffs = [0] + np.cumsum(
            [l for l in map(len, prob_prompts)]
        ).tolist()
    elif 'gpt' in model_name.lower():
        target_tok = tok(" " + target_new["str"])["input_ids"]
        inp_prompts_og = list(chain(*prob_prompts))
        inp_prompts = [
            el + tok.decode(target_tok[:i])
            for el in inp_prompts_og
            for i in range(len(target_tok))
        ]
        inp_targets = [
            tok.decode(target_tok[i])
            for _ in range(len(inp_prompts_og))
            for i in range(len(target_tok))
        ]
        stuff_probs = test_batch_prediction_acc(model, tok, inp_prompts, inp_targets)

        # Predict for neighborhood prompts (dictionary format).

        neighborhood_target_tok = tok(" " + neighborhood_prompts["target"])["input_ids"]
        neighborhood_prompts_og = neighborhood_prompts['prompt']

        neighbor_inp_prompts = [
            neighborhood_prompts_og + tok.decode(neighborhood_target_tok[:i])
            for i in range(len(neighborhood_target_tok))
        ]
        neighbor_inp_targets = [
            tok.decode(neighborhood_target_tok[i])
            for i in range(len(neighborhood_target_tok))
        ]
        neighborhood_correct = test_batch_prediction_acc(
            model,
            tok,
            neighbor_inp_prompts,
            neighbor_inp_targets
        )
        cutoffs = [0] + np.cumsum(
            [l * len(target_tok) for l in map(len, prob_prompts)]
        ).tolist()

    probs = stuff_probs + neighborhood_correct

    # Unflatten the results again into a list of lists.
    ret_probs = [probs[cutoffs[i - 1] : cutoffs[i]] for i in range(1, len(cutoffs))]
    # Structure the restuls as a dictionary.
    ret = {
        f"{key}_correct": ret_probs[i]
        for i, key in enumerate(
            [
                "rewrite_prompts",
                "paraphrase_prompts",
            ]
        )
    }
    ret["neighborhood_prompts_correct"] = neighborhood_correct

    return ret


def test_batch_prediction_acc(model, tok, prompts, target):
    prompt_tok = tok(
        prompts,
        padding=True,
        return_tensors="pt",
    ).to("cuda")

    with torch.no_grad():
        logits = model(**prompt_tok).logits
        last_non_masked = prompt_tok["attention_mask"].sum(1) - 1
        to_gather = last_non_masked.unsqueeze(1).repeat(1, logits.size(-1)).unsqueeze(1)
        gathered = torch.gather(logits, 1, to_gather).squeeze(1)
        ans = torch.argmax(gathered, dim=1)

        correct_id = tok(target, padding=True, return_tensors="pt").to("cuda")[
            "input_ids"
        ]
        # Temporary hack to deal with foreign characters.
        correct_id = correct_id[:, 0].squeeze()

        return (ans == correct_id).detach().cpu().numpy().tolist()

def test_seq2seq_batch_prediction_acc(model, tok, prompts, target, neighborhood=False):
    prompt_tok = tok(
        prompts,
        padding=True,
        return_tensors="pt",
    ).to("cuda")

    trg_tok = tok(
        [target for i in range(len(prompts))],
        padding=True,
        return_tensors="pt",
    ).to("cuda")

    prompt_tok['decoder_input_ids'] = trg_tok['input_ids']
    prompt_tok['decoder_attention_mask'] = trg_tok['attention_mask']


    with torch.no_grad():
        logits = model(**prompt_tok).logits

        assert logits.size(1) == trg_tok['input_ids'].size(1)
        ans = torch.argmax(logits, dim=-1)
        if neighborhood:
            return (trg_tok['input_ids'] == ans).squeeze().detach().cpu().numpy().tolist()

        return torch.mean((trg_tok['input_ids'] == ans).float(), dim=-1).detach().cpu().numpy().tolist()
