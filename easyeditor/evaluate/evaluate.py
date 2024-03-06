"""
Contains evaluation utilities for pytorch-based rewriting methods.
To use, simply call `compute_rewrite_quality_zsre` with the
appropriate arguments, which returns a dictionary containing them.
"""
from ..models.melo.melo import LORA

import typing
from itertools import chain
from typing import List, Optional

import numpy as np
import torch
# from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import AutoTokenizer
from ..util import HyperParams
from .portability_evaluate import compute_portability_quality
from .evaluate_utils import (
    test_seq2seq_batch_prediction_acc, 
    test_batch_prediction_acc, 
    test_prediction_acc,
    test_generation_quality, 
    PPL,
    kl_loc_loss,
    es_sent,
    es_per_icl,
    F1
)

def compute_edit_quality(
    model,
    model_name,
    hparams: HyperParams,
    tok: AutoTokenizer,
    record: typing.Dict,
    device,
    eval_metric: str = 'token_em',
    test_generation = False
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
    if isinstance(model,LORA):
        model=model.model
    # First, unpack rewrite evaluation record.
    target_new, ground_truth = (
        record[x] for x in ["target_new", "ground_truth"]
    )

    rewrite_prompts = record["prompt"]
    rephrase_prompts = record["rephrase_prompt"] if 'rephrase_prompt' in record.keys() else None
    ret = compute_rewrite_or_rephrase_quality(model, model_name, hparams, tok,
                                              rewrite_prompts, target_new, device=device, eval_metric=eval_metric)

    ret['locality'] = {}
    ret['portability'] = {}
    if rephrase_prompts is not None:
        ret.update(
            compute_rewrite_or_rephrase_quality(model, model_name, hparams, tok,
                                                rephrase_prompts, target_new, device=device, test_rephrase=True, eval_metric=eval_metric)
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
    if test_generation:
        if hparams.alg_name == 'GRACE':
            ret['fluency'] = test_generation_quality(model=model,tok=tok,prefixes=rewrite_prompts if isinstance(rewrite_prompts,list) else [rewrite_prompts,], max_out_len=100, vanilla_generation=True)
        else:
            ret['fluency'] = test_generation_quality(model=model,tok=tok,prefixes=rewrite_prompts if isinstance(rewrite_prompts,list) else [rewrite_prompts,], max_out_len=100, vanilla_generation=False)
    return ret

def compute_rewrite_or_rephrase_quality(
    model,
    model_name,
    hparams: HyperParams,
    tok: AutoTokenizer,
    prompt: str,
    target_new: str,
    device,
    test_rephrase: bool = False,
    eval_metric: str = 'token_em'
) -> typing.Dict:
    
    if not test_rephrase:
        key = 'rewrite'
    else:
        key = 'rephrase'
    if eval_metric == 'ppl':
        ppl = PPL(model, tok, prompt, target_new, device)
        ret = {
            f"{key}_ppl": ppl
        }
    elif hparams.alg_name=="GRACE":
        # ppl = PPL(model, tok, prompt, target_new, device)
        if 't5' in model_name.lower():
            acc = test_seq2seq_batch_prediction_acc(model, tok, hparams, prompt, target_new, device)
        else:
            acc = test_prediction_acc(model, tok, hparams, prompt, target_new, device, vanilla_generation=True)
        f1 = F1(model,tok,hparams,prompt,target_new,device, vanilla_generation=True)
        ret = {
            f"{key}_acc": acc,
            # f"{key}_PPL": ppl,
            f"{key}_F1":f1     
        }        
    else:
        if 't5' in model_name.lower():
            acc = test_seq2seq_batch_prediction_acc(model, tok, hparams, prompt, target_new, device)
        else:
            acc = test_prediction_acc(model, tok, hparams, prompt, target_new, device)
        ret = {
            f"{key}_acc": acc
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
        loc_tokens = test_seq2seq_batch_prediction_acc(model, tok, hparams, prompt, locality_ground_truth, device, locality=True)
    else:
        loc_tokens = test_prediction_acc(model, tok, hparams, prompt, locality_ground_truth, device, locality=True, vanilla_generation=hparams.alg_name=='GRACE')

    if type(loc_tokens) is not list:
        loc_tokens = [loc_tokens,]

    ret = {
        f"{locality_key}_output": loc_tokens
    }
    return ret

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
    :param snips: ???
    :param vec: ???
    :return: Dictionary containing rewriting metrics
    """

    # First, unpack rewrite evaluation record.
    target_new, ground_truth = (
        record[x] for x in ["target_new", "ground_truth"]
    )
    prompt = record["prompt"]
    rephrase = record["rephrase_prompt"] if 'rephrase_prompt' in record.keys() else None
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
            if isinstance(record['locality'][locality_key]['ground_truth'],list):
                pre_neighbor = []
                post_neighbor = []
                for x_a, x_p in zip(record['locality'][locality_key]['ground_truth'],record['locality'][locality_key]['prompt']):             
                    tmp_pre_neighbor = icl_lm_eval(model, model_name, hparams, tok, [''], x_a,
                                            f"New Fact: {prompt} {target_new}\nPrompt: {x_p}", neighborhood=True)
                    tmp_post_neighbor = icl_lm_eval(model, model_name, hparams, tok, icl_examples, x_a,
                                                f"New Fact: {prompt} {target_new}\nPrompt: {x_p}", neighborhood=True)
                    if type(tmp_pre_neighbor) is not list:
                        tmp_pre_neighbor = [tmp_pre_neighbor, ]
                    if type(tmp_post_neighbor) is not list:
                        tmp_post_neighbor = [tmp_post_neighbor, ]
                    assert len(tmp_pre_neighbor) == len(tmp_post_neighbor)
                    pre_neighbor.append(tmp_pre_neighbor)
                    post_neighbor.append(tmp_post_neighbor)
                res = []
                for ans,label in zip(pre_neighbor,post_neighbor):
                    temp_acc = np.mean(np.equal(ans, label))
                    if np.isnan(temp_acc):
                        continue
                    res.append(temp_acc)
                ret['locality'][f'{locality_key}_acc'] = res
            else:
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
                icl_input = ['']
                x_prefix=""
            else:
                icl_input = icl_examples
                x_prefix=f"New Fact: {prompt} {target_new}\nPrompt: "
            if isinstance(record['portability'][portability_key]['ground_truth'],list):
                portability_acc = []
                for x_a, x_p in zip(record['portability'][portability_key]['ground_truth'],record['portability'][portability_key]['prompt']): 
                    tmp_portability_acc = icl_lm_eval(model, model_name, hparams, tok,icl_input, x_a,
                                            f"{x_prefix}{x_p}")
                portability_acc.append(tmp_portability_acc)
            else:
                portability_acc = icl_lm_eval(model, model_name, hparams, tok, [''], record['portability'][portability_key]['ground_truth'],
                                                record['portability'][portability_key]['prompt'])
                portability_acc = icl_lm_eval(model, model_name, hparams, tok, icl_examples, record['portability'][portability_key]['ground_truth'],
                                                f"New Fact: {prompt} {target_new}\nPrompt: {record['portability'][portability_key]['prompt']}")
            ret['portability'][f'{portability_key}_acc'] = portability_acc
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
    elif 'llama' in model_name.lower():
        target_ids = tokenizer(target, return_tensors='pt')['input_ids'].to(device)
        encodings = tokenizer(''.join(icl_examples) + f'{x} {target}', return_tensors='pt')
        input_ids = encodings['input_ids'].to(device)
        attention_mask = encodings['attention_mask'].to(device)
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
        ans = torch.argmax(logits, dim=-1)[:,-target_ids.size(1):-1].squeeze()
        target_ids = target_ids[:,1:]   
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

def compute_icl_multimodal_edit_quality(
    model,
    model_name,
    hparams: HyperParams,
    tok: AutoTokenizer,
    # vis_tok,
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
    :param snips: ???
    :param vec: ???
    :return: Dictionary containing rewriting metrics
    """
    vis_root = hparams.coco_image
    rephrase_root = hparams.rephrase_image
    # First, unpack rewrite evaluation record.
    target = record["target"]
    prompt = record["prompt"]
    image = record["image"] if record["image"].is_cuda else record["image"].to(hparams.device)
    rephrase = record["rephrase_prompt"] if 'rephrase_prompt' in record.keys() else None
    rephrase_image = record["image_rephrase"] if 'image_rephrase' in record.keys() else None
    if rephrase_image is not None:
        rephrase_image = rephrase_image if rephrase_image.is_cuda else rephrase_image.to(hparams.device)
    
    if "locality_prompt" in record.keys():
        loc_q = record["locality_prompt"]
        loc_a = record["locality_ground_truth"]
    if "multimodal_locality_image" in record.keys():
        m_loc_image = record["multimodal_locality_image"] if record["multimodal_locality_image"].is_cuda else record["multimodal_locality_image"].to(hparams.device)
        m_loc_q = record["multimodal_locality_prompt"]
        m_loc_a = record["multimodal_locality_ground_truth"]
    
    new_fact = f'New Fact: {prompt} {target}\nPrompt: {prompt}'

    if pre_edit:
        edit_acc, _ = icl_multimodal_lm_eval(model, model_name, hparams, tok, icl_examples,
                                       target, prompt, image)
    else:
        edit_acc, _ = icl_multimodal_lm_eval(model, model_name, hparams, tok, icl_examples,
                                              target, new_fact, image)
    ret = {
        f"rewrite_acc": edit_acc
    }
    if rephrase is not None:
        rephrase_acc, _ = icl_multimodal_lm_eval(model, model_name, hparams, tok, icl_examples,
                               target, f'New Fact: {prompt} {target}\nPrompt: {rephrase}', image)
        ret['rephrase_acc'] = rephrase_acc
        
    if "image_rephrase" in record.keys():
        rephrase_image_acc, _ = icl_multimodal_lm_eval(model, model_name, hparams, tok, icl_examples,
                               target, new_fact, rephrase_image)
        ret['rephrase_image_acc'] = rephrase_image_acc
    
    if "locality_prompt" in record.keys():
        if pre_edit:
            _, _, locality_output = icl_multimodal_lm_eval(model, model_name, hparams, tok, icl_examples,
                                    loc_a, loc_q, None, is_loc=True) 
        else:
            _, _, locality_output = icl_multimodal_lm_eval(model, model_name, hparams, tok, icl_examples,
                                    loc_a, f'New Fact: {prompt} {target}\nPrompt: {loc_q}', None, is_loc=True) 
        ret['locality_output'] = locality_output
    
    if "multimodal_locality_image" in record.keys():
        if pre_edit:
            _, _, locality_image_output = icl_multimodal_lm_eval(model, model_name, hparams, tok, icl_examples,
                                    m_loc_a, m_loc_q, m_loc_image, is_loc=True) 
        else:
            _, _, locality_image_output = icl_multimodal_lm_eval(model, model_name, hparams, tok, icl_examples,
                                    m_loc_a, f'New Fact: {prompt} {target}\nPrompt: {m_loc_q}', m_loc_image, is_loc=True) 
        ret['multimodal_locality_output'] = locality_image_output
            
    return ret

def icl_multimodal_lm_eval(
        model,
        model_name,
        hparams: HyperParams,
        tokenizer,
        icl_examples,
        target,
        x,
        image,
        is_loc=False,
        neighborhood=False
)-> typing.Dict:
    device = torch.device(f'cuda:{hparams.device}')
    
    samples = prepare_multimodal_edit(hparams, tokenizer, target, [''.join(icl_examples) + f'{x}'], image) 
    
    return compute_multimodal_edit_quality(model, samples) if not is_loc else compute_multimodal_edit_quality_demo(model, samples)

def prepare_multimodal_edit(hparams,
                            tok,
                            target,
                            prompts,
                            image):
    if isinstance(target, str):
        target = [target,]
    if isinstance(prompts, str):
        prompts = [prompts,]
    if image is not None and len(image.shape) == 3:
        image = image.unsqueeze(0)
    text_input = [prompt_ + ' ' + target_ for prompt_, target_ in zip(prompts, target)]
    
    if hparams.model_name == 'minigpt4':
        prompts_len = [len(tok.encode(prompt, add_special_tokens=False)) for prompt in prompts]
        target = tok(target, add_special_tokens=False, return_tensors="pt",)["input_ids"]
    else:
        prompts_len = [len(tok.encode(prompt,  add_special_tokens=False)) for prompt in prompts]  
        target = tok([' ' + target_ if target_[0] != ' ' else target_ for target_ in target], add_special_tokens=False, return_tensors="pt",)["input_ids"]
        
    ret = {
        'text_input': text_input,
        'image': image,
        'labels': target,
        'prompts_len': prompts_len        
    } 
    return ret

def compute_multimodal_edit_quality(model, batch):
    
    with torch.no_grad():
        outputs = model(batch)
        if isinstance(outputs, torch.Tensor):
            logits = outputs.detach().cpu()
        else:
            logits = outputs.logits.detach().cpu()    
        # targ = outputs.labels.detach().cpu()
        targ = batch["labels"].cpu()
    if logits.dim() == 3:
        logits = logits[:, :-1]
        # targ = targ[:, 1:]
        logits = logits[:, -targ.shape[1]:]
    mask = targ != -100
    targ[~mask] = 0
    pred_ids = logits.argmax(-1).masked_fill(~mask, 0).detach().cpu()
    correct = pred_ids == targ
    correct = correct & mask
    num_non_padding = mask.sum().float().item()
    acc = correct.sum() / num_non_padding
    
    return acc, pred_ids.numpy()
  
def compute_multimodal_edit_quality_demo(model, batch):
    
    with torch.no_grad():
        outputs = model(batch)
        if isinstance(outputs, torch.Tensor):
            logits = outputs.detach().cpu()
        else:
            logits = outputs.logits.detach().cpu()    
        # targ = outputs.labels.detach().cpu()
        targ = batch["labels"].cpu()
    logits_ = logits.clone()
    if logits.dim() == 3:
        logits = logits[:, :-1]
        # targ = targ[:, 1:]
        logits = logits[:, -targ.shape[1]:]
    mask = targ != -100
    targ[~mask] = 0
    pred_ids = logits.argmax(-1).masked_fill(~mask, 0).detach().cpu()
    correct = pred_ids == targ
    correct = correct & mask
    num_non_padding = mask.sum().float().item()
    acc = correct.sum() / num_non_padding
    
    return acc, pred_ids.numpy(), logits_

def compute_multimodal_edit_results(
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
    ret = {}
    # First, unpack rewrite evaluation record.
    
    target = record["target"]
    rewrite_prompts = record["prompt"]
    image = record["image"]
    
    edit_inner = prepare_multimodal_edit(hparams, tok, target, rewrite_prompts, image)
    ret['rewrite_acc'], _ = compute_multimodal_edit_quality(model, edit_inner)
    
    if "rephrase_prompt" in record.keys():
        rephrase_prompts = record["rephrase_prompt"]
        edit_outer = prepare_multimodal_edit(hparams, tok, target, rephrase_prompts, image)
        ret['rephrase_acc'], _ = compute_multimodal_edit_quality(model, edit_outer)
        
    if "image_rephrase" in record.keys():
        rephrase_image = record["image_rephrase"]
        edit_image_outer = prepare_multimodal_edit(hparams, tok, target, rewrite_prompts, rephrase_image) 
        ret['image_rephrase_acc'], _ = compute_multimodal_edit_quality(model, edit_image_outer)

    if 'locality_prompt' in record.keys():
        locality_prompt = record["locality_prompt"]
        locality_ground_truth = record["locality_ground_truth"]
        locality = prepare_multimodal_edit(hparams, tok, locality_ground_truth, locality_prompt, None)
        _, ret['locality_output'] = compute_multimodal_edit_quality(model, locality)
        
    if 'multimodal_locality_prompt' in record.keys():
        m_loc_prompt = record["multimodal_locality_prompt"]
        m_loc_ground_truth = record["multimodal_locality_ground_truth"]
        m_loc_image = record["multimodal_locality_image"]
        m_locality = prepare_multimodal_edit(hparams, tok, m_loc_ground_truth, m_loc_prompt, m_loc_image)
        _, ret['multimodal_locality_output'] = compute_multimodal_edit_quality(model, m_locality)
    # Form a list of lists of prefixes to test.

    return ret
  
def compute_multimodal_edit_results_demo(
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
    ret = {}
    # First, unpack rewrite evaluation record.
    
    target = record["target"]
    rewrite_prompts = record["prompt"]
    image = record["image"]
    
    edit_inner = prepare_multimodal_edit(hparams, tok, target, rewrite_prompts, image)
    ret['rewrite_acc'], _, logits = compute_multimodal_edit_quality_demo(model, edit_inner)
    
    if "rephrase_prompt" in record.keys():
        rephrase_prompts = record["rephrase_prompt"]
        edit_outer = prepare_multimodal_edit(hparams, tok, target, rephrase_prompts, image)
        ret['rephrase_acc'], _ = compute_multimodal_edit_quality(model, edit_outer)
        
    if "image_rephrase" in record.keys():
        rephrase_image = record["image_rephrase"]
        edit_image_outer = prepare_multimodal_edit(hparams, tok, target, rewrite_prompts, rephrase_image) 
        ret['image_rephrase_acc'], _ = compute_multimodal_edit_quality(model, edit_image_outer)

    if 'locality_prompt' in record.keys():
        locality_prompt = record["locality_prompt"]
        locality_ground_truth = record["locality_ground_truth"]
        locality = prepare_multimodal_edit(hparams, tok, locality_ground_truth, locality_prompt, None)
        _, ret['locality_output'] = compute_multimodal_edit_quality(model, locality)
        
    if 'multimodal_locality_prompt' in record.keys():
        m_loc_prompt = record["multimodal_locality_prompt"]
        m_loc_ground_truth = record["multimodal_locality_ground_truth"]
        m_loc_image = record["multimodal_locality_image"]
        m_locality = prepare_multimodal_edit(hparams, tok, m_loc_ground_truth, m_loc_prompt, m_loc_image)
        _, ret['multimodal_locality_output'] = compute_multimodal_edit_quality(model, m_locality)
    # Form a list of lists of prefixes to test.

    return ret, logits


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

def compute_sent_metric(
    model,
    edited_model,
    model_name,
    hparams: HyperParams,
    tok: AutoTokenizer,
    metric_kwargs: typing.Dict,
    device,
    test_generation=True
    ):
    
    if "llama" not in model_name:
        raise NotImplementedError("currently only support for llama")
        
    def get_edit_labels(ids, prompts=None):
        labels = ids.clone()
        labels[labels == tok.pad_token_id] = -100
        return labels
        
    same_mask = torch.tensor([i == o for i, o in zip(metric_kwargs["inner_target"], metric_kwargs["all_target"])], device=device)
    edit_toks = {
        f"{k1}_{k2}": v2.to(device)
        for k1, v1 in {
            "inner": metric_kwargs["inner_all_qa"],
            "outer": metric_kwargs["outer_all_qa"],
        }.items()
        for k2, v2 in tok(
            v1,
            return_tensors="pt",
            padding=True,
            max_length=128,
            truncation=True,
        ).items()
    }
    for key in ["inner", "outer"]:
        value = edit_toks[f"{key}_input_ids"]
        mask = [([True] * value.shape[-1])] * value.shape[0]
        for i in range(value.shape[0]):
            sep_idx = list(value[i]).index(tok.convert_tokens_to_ids("</s>"))
            for j in range(sep_idx): #连带</s>一块mask掉
                mask[i][j] = False
        edit_toks[key + "_q_mask"] = torch.tensor(mask).to(device)

    with torch.no_grad():
        inner_base_logits = model(
            input_ids=edit_toks["inner_input_ids"],
            attention_mask=edit_toks["inner_attention_mask"],   
        )["logits"]
        inner_edit_logits = edited_model(
            input_ids=edit_toks["inner_input_ids"],
            attention_mask=edit_toks["inner_attention_mask"],   
        )["logits"]
        
        outer_base_logits = model(
            input_ids=edit_toks["outer_input_ids"],
            attention_mask=edit_toks["outer_attention_mask"],   
        )["logits"]
        outer_edit_logits = edited_model(
            input_ids=edit_toks["outer_input_ids"],
            attention_mask=edit_toks["outer_attention_mask"],   
        )["logits"]
    
    result = {
        "es": es_sent(inner_base_logits, inner_edit_logits, edit_toks["inner_q_mask"], get_edit_labels(edit_toks["inner_input_ids"]), same_mask).item(),
        "dd": kl_loc_loss(outer_base_logits, outer_edit_logits, edit_toks["outer_q_mask"]).item(),
    }
    if  test_generation:
        result['fluency'] = test_generation_quality(model=model,tok=tok,prefixes=metric_kwargs["inner_q"] if isinstance(metric_kwargs["inner_q"],list) else [metric_kwargs["inner_q"],], max_out_len=100)
    return result


def compute_per_ike_metric(
    example,
    model,
    tok,
    test_generation=False,
):
    with torch.no_grad():
        inner_pre_logits = model(
            input_ids=example["inner_pre_prompt"]["input_ids"],
            attention_mask=example["inner_pre_prompt"]["attention_mask"],   
            labels=example["inner_pre_prompt"]["labels"],
        )["logits"]
        inner_edit_logits = model(
            input_ids=example["inner_edit_prompt"]["input_ids"],
            attention_mask=example["inner_edit_prompt"]["attention_mask"],   
            labels=example["inner_edit_prompt"]["labels"],
        )["logits"]

        outer_pre_logits = model(
            input_ids=example["outer_pre_prompt"]["input_ids"],
            attention_mask=example["outer_pre_prompt"]["attention_mask"],   
            labels=example["outer_pre_prompt"]["labels"],
        )["logits"]

        outer_edit_logits = model(
            input_ids=example["outer_edit_prompt"]["input_ids"],
            attention_mask=example["outer_edit_prompt"]["attention_mask"],   
            labels=example["outer_edit_prompt"]["labels"],
        )["logits"]
        
        result = {
            "es": es_per_icl(example, inner_pre_logits, inner_edit_logits)["acc_per"].item(),
            "dd": kl_loc_loss(outer_pre_logits, outer_edit_logits, example["outer_pre_prompt"]["q_mask"]).item()
        }
        
    return result
    