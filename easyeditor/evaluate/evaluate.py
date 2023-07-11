"""
Contains evaluation utilities for pytorch-based rewriting methods.
To use, simply call `compute_rewrite_quality_zsre` with the
appropriate arguments, which returns a dictionary containing them.
"""

import typing
from itertools import chain
from typing import List

import numpy as np
import torch
# from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import AutoTokenizer
from ..util import HyperParams
from .portability_evaluate import compute_portability_quality


def compute_icl_edit_quality(
    model,
    model_name,
    hparams: HyperParams,
    tok: AutoTokenizer,
    icl_examples,
    record: typing.Dict,
    device,
    pre_edit: bool = False
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
    target_new, ground_truth = (
        record[x] for x in ["target_new", "ground_truth"]
    )
    prompt = record["prompt"]
    rephrase = record["rephrase_prompt"] if 'rephrase_prompt' in record.keys() else None
    # locality_prompt = record["locality_prompt"] if 'locality_prompt' in record.keys() else None
    # locality_ground_truth = record["locality_ground_truth"] if 'locality_ground_truth' in record.keys() else None

    # one_hop_prompt = record["one_hop_prompt"] if 'one_hop_prompt' in record.keys() else None
    # one_hop_ground_truth = record["one_hop_ground_truth"] if 'one_hop_ground_truth' in record.keys() else None
    # synonym_prompt = record["synonym_prompt"] if 'synonym_prompt' in record.keys() else None
    # synonym_ground_truth = record["synonym_ground_truth"] if 'synonym_ground_truth' in record.keys() else None
    # inverse_relation_prompt = record["inverse_relation_prompt"] if 'inverse_relation_prompt' in record.keys() else None
    # inverse_relation_ground_truth = record["inverse_relation_ground_truth"] if 'inverse_relation_ground_truth' in record.keys() else None

    new_fact = f'New Fact: {prompt} {target_new}\nPrompt: {prompt}'

    if pre_edit:
        edit_acc = icl_lm_eval(model, model_name, hparams, tok, icl_examples,
                                       target_new, prompt)
    else:
        edit_acc = icl_lm_eval(model, model_name, hparams, tok, icl_examples,
                                              target_new, new_fact)
    ret = {
        f"rewrite_acc": edit_acc
    }
    ret['locality'] = {}
    ret['portability'] = {}
    if rephrase is not None:
        rephrase_acc = icl_lm_eval(model, model_name, hparams, tok, icl_examples,
                               target_new, f'New Fact: {prompt} {target_new}\nPrompt: {rephrase}')
        ret['rephrase_acc'] = rephrase_acc

    if 'locality' in record.keys() and any(record['locality']):
        for locality_key in record['locality'].keys():
            pre_neighbor = icl_lm_eval(model, model_name, hparams, tok, [''], record['locality'][locality_key]['ground_truth'],
                                       f"New Fact: {prompt} {target_new}\nPrompt: {record['locality'][locality_key]['prompt']}", neighborhood=True)
            post_neighbor = icl_lm_eval(model, model_name, hparams, tok, icl_examples, record['locality'][locality_key]['ground_truth'],
                                        f"New Fact: {prompt} {target_new}\nPrompt: {record['locality'][locality_key]['prompt']}", neighborhood=True)
            if type(pre_neighbor) is not list:
                pre_neighbor = [pre_neighbor, ]
            if type(post_neighbor) is not list:
                post_neighbor = [post_neighbor, ]
            assert len(pre_neighbor) == len(post_neighbor)

            ret['locality'][f'{locality_key}_acc'] = np.mean(np.equal(pre_neighbor, post_neighbor))
    # Form a list of lists of prefixes to test.
    if 'portability' in record.keys() and any(record['portability']):
        for portability_key in record['portability'].keys():
            if pre_edit:
                portability_acc = icl_lm_eval(model, model_name, hparams, tok, icl_examples, record['portability'][portability_key]['ground_truth'],
                                              record['portability'][portability_key]['prompt'])
            else:
                portability_acc = icl_lm_eval(model, model_name, hparams, tok, icl_examples, record['portability'][portability_key]['ground_truth'],
                                              f"New Fact: {prompt} {target_new}\nPrompt: {record['portability'][portability_key]['prompt']}")
            ret['portability'][f'{portability_key}_acc'] = portability_acc
    # if one_hop_prompt is not None:
    #     one_hop_acc = icl_lm_eval(model, model_name, hparams, tok, icl_examples,
    #                            one_hop_ground_truth, f'New Fact: {prompt} {target_new}\nPrompt: {one_hop_prompt}')
    #     ret['one_hop_acc'] = one_hop_acc
    # if synonym_prompt is not None:
    #     synonym_acc = icl_lm_eval(model, model_name, hparams, tok, icl_examples,
    #                            synonym_ground_truth, f'New Fact: {prompt} {target_new}\nPrompt: {synonym_prompt}')
    #     ret['synonym_acc'] = synonym_acc
    # if inverse_relation_prompt is not None:
    #     inverse_relation_acc = icl_lm_eval(model, model_name, hparams, tok, icl_examples,
    #                            inverse_relation_ground_truth, f'New Fact: {prompt} {target_new}\nPrompt: {inverse_relation_prompt}')
    #     ret['inverse_relation_acc'] = inverse_relation_acc
    return ret

def icl_lm_eval(
        model,
        model_name,
        hparams: HyperParams,
        tokenizer,
        icl_examples,
        target,
        x,
        neighborhood=False
)-> typing.Dict:
    device = torch.device(f'cuda:{hparams.device}')
    if 't5' in model_name.lower():
        target_len = len(tokenizer.encode(target))
        target_ids = tokenizer(f'{x} {target}', return_tensors='pt')['input_ids'].to(device)
        encodings = tokenizer(''.join(icl_examples), return_tensors='pt')
        input_ids = encodings['input_ids'].to(device)
        attention_mask = encodings['attention_mask'].to(device)
        with torch.no_grad():
            logits = model(input_ids=input_ids, attention_mask=attention_mask, labels=target_ids).logits
            ans = torch.argmax(logits, dim=-1)[:,-target_len:-1].squeeze()
            target_ids = target_ids[:,-target_len:-1]
            if neighborhood:
                return ans.squeeze().detach().cpu().numpy().tolist()
            return torch.mean((ans == target_ids.to(ans.device).squeeze()).float(), dim=-1).detach().cpu().numpy().tolist()
    else:
        target_ids = tokenizer(' ' + target + '\n', return_tensors='pt')['input_ids'].to(device)
        encodings = tokenizer(''.join(icl_examples) + f'{x} {target}', return_tensors='pt')
        input_ids = encodings['input_ids'].to(device)
        attention_mask = encodings['attention_mask'].to(device)
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
        ans = torch.argmax(logits, dim=-1)[:,-target_ids.size(1):-1].squeeze()
        target_ids = target_ids[:,:-1]
        if neighborhood:
            return ans.squeeze().detach().cpu().numpy().tolist()
        return torch.mean((ans == target_ids.to(ans.device).squeeze()).float(), dim=-1).detach().cpu().numpy().tolist()

# TODO: Support GPT Evaluation(predict token one by one)
def compute_rewrite_or_rephrase_quality(
    model,
    model_name,
    hparams: HyperParams,
    tok: AutoTokenizer,
    prompt: str,
    target_new: str,
    device,
    test_rephrase: bool = False
) -> typing.Dict:

    if 't5' in model_name.lower():
        stuff_probs = test_seq2seq_batch_prediction_acc(model, tok, hparams,
                                                        prompt,
                                                        target_new,
                                                        device)
    elif 'gpt' in model_name.lower():
        target_tok = tok(target_new, truncation=True, max_length=hparams.max_length)["input_ids"]
        inp_prompts = [prompt]
        inp_prompts.extend([
            prompt + ' ' + tok.decode(target_tok[:i])
            for i in range(1, len(target_tok))
        ])
        # inp_targets = [
        #     tok.decode(target_tok[i])
        #     for i in range(len(target_tok))
        # ]
        stuff_probs = test_batch_prediction_acc(model, tok, hparams, inp_prompts, target_tok, device)
    elif 'llama' in model_name.lower():
        target_tok = tok(target_new, truncation=True, max_length=hparams.max_length)["input_ids"][1:] #erase bos_token_id
        inp_prompts = [prompt]
        inp_prompts.extend([
            prompt + ' ' + tok.decode(target_tok[:i])
            for i in range(1, len(target_tok))
        ])
        stuff_probs = test_batch_prediction_acc(model, tok, hparams, inp_prompts, target_tok, device)

    probs = stuff_probs

    # Structure the restuls as a dictionary.

    if not test_rephrase:
        key = 'rewrite'
    else:
        key = 'rephrase'
    ret = {
        f"{key}_acc": probs
    }
    return ret

def compute_locality_quality(
    model,
    model_name,
    hparams: HyperParams,
    tok: AutoTokenizer,
    locality_key: str,
    prompt: str,
    locality_ground_truth: str,
    device,
) -> typing.Dict:

    if 't5' in model_name.lower():
        locality_correct = test_seq2seq_batch_prediction_acc(model, tok, hparams,
                                                                 prompt,
                                                                 locality_ground_truth,
                                                                 device,
                                                                 locality=True)
    elif 'gpt' in model_name.lower():
        target_tok = tok(locality_ground_truth, truncation=True, max_length=hparams.max_length)["input_ids"]
        inp_prompts = [prompt]
        inp_prompts.extend([
            prompt + ' ' + tok.decode(target_tok[:i])
            for i in range(1, len(target_tok))
        ])

        locality_correct = test_batch_prediction_acc(model, tok, hparams, inp_prompts, target_tok, device, locality=True)
    elif 'llama' in model_name.lower():
        target_tok = tok(locality_ground_truth, truncation=True, max_length=hparams.max_length)["input_ids"][1:] # erase bos_token_id
        inp_prompts = [prompt]
        inp_prompts.extend([
            prompt + ' ' + tok.decode(target_tok[:i])
            for i in range(1, len(target_tok))
        ])
        # inp_targets = [
        #     tok.decode(target_tok[i])
        #     for i in range(len(target_tok))
        # ]

        locality_correct = test_batch_prediction_acc(model, tok, hparams, inp_prompts, target_tok, device, locality=True)

    probs = locality_correct

    if type(probs) is not list:
        probs = [probs,]

    ret = {
        f"{locality_key}_output": probs
    }
    return ret


def compute_edit_quality(
    model,
    model_name,
    hparams: HyperParams,
    tok: AutoTokenizer,
    record: typing.Dict,
    device
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
    target_new, ground_truth = (
        record[x] for x in ["target_new", "ground_truth"]
    )

    rewrite_prompts = record["prompt"]
    rephrase_prompts = record["rephrase_prompt"] if 'rephrase_prompt' in record.keys() else None

    # locality_prompts = record["locality_prompt"] if 'locality_prompt' in record.keys() else None
    # locality_ground_truth = record["locality_ground_truth"] if 'locality_ground_truth' in record.keys() else None
    #
    # one_hop_prompt = record["one_hop_prompt"] if 'one_hop_prompt' in record.keys() else None
    # one_hop_ground_truth = record["one_hop_ground_truth"] if 'one_hop_ground_truth' in record.keys() else None
    # synonym_prompt = record["synonym_prompt"] if 'synonym_prompt' in record.keys() else None
    # synonym_ground_truth = record["synonym_ground_truth"] if 'synonym_ground_truth' in record.keys() else None
    # inverse_relation_prompt = record["inverse_relation_prompt"] if 'inverse_relation_prompt' in record.keys() else None
    # inverse_relation_ground_truth = record["inverse_relation_ground_truth"] if 'inverse_relation_ground_truth' in record.keys() else None

    ret = compute_rewrite_or_rephrase_quality(model, model_name, hparams, tok,
                                              rewrite_prompts, target_new, device=device)

    ret['locality'] = {}
    ret['portability'] = {}
    if rephrase_prompts is not None:
        ret.update(
            compute_rewrite_or_rephrase_quality(model, model_name, hparams, tok,
                                                rephrase_prompts, target_new, device=device, test_rephrase=True)
        )

    if 'locality' in record.keys() and any(record['locality']):
        for locality_key in record['locality'].keys():
            ret['locality'].update(
                compute_locality_quality(model, model_name, hparams, tok, locality_key,
                                         record['locality'][locality_key]['prompt'],
                                         record['locality'][locality_key]['ground_truth'], device=device)
            )
    if 'portability' in record.keys() and any(record['portability']):
        for portability_key in record['portability'].keys():
            ret['portability'].update(
                compute_portability_quality(model, model_name, hparams, tok, portability_key,
                                            record['portability'][portability_key]['prompt'],
                                            record['portability'][portability_key]['ground_truth'], device=device)
            )
    # Form a list of lists of prefixes to test.

    return ret


def test_batch_prediction_acc(model, tok, hparams, prompts, target, device, locality=False):
    prompt_tok = tok(
        prompts,
        padding=True,
        truncation=True,
        max_length=hparams.max_length,
        return_tensors="pt",
    ).to(f"cuda:{device}")

    with torch.no_grad():
        outputs = model(**prompt_tok)
        if type(outputs) is torch.Tensor:
            logits = outputs
        else:
            logits = outputs.logits

        if tok.padding_side == 'left':
            ans = torch.argmax(logits, dim=-1)[:, -1].squeeze()
        else:
            last_non_masked = prompt_tok["attention_mask"].sum(1) - 1
            to_gather = last_non_masked.unsqueeze(1).repeat(1, logits.size(-1)).unsqueeze(1)
            gathered = torch.gather(logits, 1, to_gather).squeeze(1)
            ans = torch.argmax(gathered, dim=1)

        # correct_id = tok(target, padding=True, truncation=True, max_length=hparams.max_length, return_tensors="pt").to(f"cuda:{device}")[
        #     "input_ids"
        # ]
        # Temporary hack to deal with foreign characters.
        # correct_id = correct_id[:, -1].squeeze()
        ans = ans.squeeze().detach().cpu().numpy().tolist()

        if locality:
            return ans

        return np.mean(np.equal(ans, target))

def test_seq2seq_batch_prediction_acc(model, tok, hparams, prompt, target, device, locality=False):
    prompt_tok = tok(
        prompt,
        padding=True,
        truncation=True,
        max_length=hparams.max_length,
        return_tensors="pt",
    ).to(f"cuda:{device}")

    trg_tok = tok(
        target,
        padding=True,
        truncation=True,
        max_length=hparams.max_length,
        return_tensors="pt",
    ).to(f"cuda:{device}")

    prompt_tok['labels'] = trg_tok['input_ids']
    # prompt_tok['decoder_attention_mask'] = trg_tok['attention_mask']


    with torch.no_grad():
        outputs = model(**prompt_tok)
        if type(outputs) is torch.Tensor:
            logits = outputs
        else:
            logits = outputs.logits

        assert logits.size(1) == trg_tok['input_ids'].size(1)
        ans = torch.argmax(logits, dim=-1)
        if locality:
            return ans.squeeze().detach().cpu().numpy().tolist()

        return torch.mean((trg_tok['input_ids'][:,:-1] == ans[:,:-1]).float(), dim=-1).detach().cpu().numpy().tolist()[0]
