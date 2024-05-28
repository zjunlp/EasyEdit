from sentence_transformers import SentenceTransformer, util
from transformers import AutoModelForCausalLM, AutoTokenizer
import pickle
import json
from torch.utils.data import Dataset
from .ike_hparams import IKEHyperParams, IKEMultimodalHyperParams
import os
from copy import deepcopy
from typing import Any, Dict, List, Tuple

import torch
from torch import tensor


def apply_ike_to_model(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    request: Dict,
    hparams: IKEHyperParams,
    copy=False,
    return_orig_weights=False,
    keep_original_weight=False,
    train_ds=None,
    **kwargs: Any,
) -> Tuple[AutoModelForCausalLM, Dict[str, Any]]:

    if type(request) is list:
        request = request[0]

    assert train_ds is not None
    device = torch.device(f'cuda:{hparams.device}')
    sentence_model = SentenceTransformer(hparams.sentence_model_name).to(device)

    safe_model_name = hparams.sentence_model_name.rsplit('/', 1)[-1]
    with open(f'{hparams.results_dir}/{hparams.alg_name}/embedding/'
              f'{safe_model_name}_{type(train_ds).__name__}_{len(train_ds)}.pkl', "rb") as fIn:
        stored_data = pickle.load(fIn)
        stored_sentences = stored_data['sentences']
        stored_embeddings = stored_data['embeddings']
    stored_embeddings = torch.tensor(stored_embeddings).to(device)
    stored_embeddings = util.normalize_embeddings(stored_embeddings)

    new_fact = request['prompt'] + ' ' + request['target_new']
    query_sentence = f"New Fact: {new_fact}\nPrompt: {request['prompt']}\n\n"
    query_embedding = util.normalize_embeddings(torch.tensor(sentence_model.encode(
        query_sentence, show_progress_bar=False)).unsqueeze(0).to(device))

    hits = util.semantic_search(query_embedding, stored_embeddings, score_function=util.dot_score, top_k=hparams.k)
    assert len(hits) == 1
    hit = hits[0]
    icl_examples = [stored_sentences[hit[k]["corpus_id"]] for k in range(len(hit))]
    icl_examples.append(f'New Fact: {new_fact}\nPrompt: {new_fact}\n\n')

    return icl_examples

def apply_ike_to_multimodal_model(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    request: Dict,
    hparams: IKEMultimodalHyperParams,
    copy=False,
    return_orig_weights=False,
    keep_original_weight=False,
    train_ds=None,
    **kwargs: Any,    
) -> Tuple[AutoModelForCausalLM, Dict[str, Any]]:
    
    assert train_ds is not None
    device = torch.device(f'cuda:{hparams.device}')
    sentence_model = SentenceTransformer(hparams.sentence_model_name).to(device)

    safe_model_name = hparams.sentence_model_name.rsplit('/', 1)[-1]
    with open(f'{hparams.results_dir}/{hparams.alg_name}/embedding/'
              f'{hparams.task_name}_embeddings.pkl', "rb") as fIn:
        stored_data = pickle.load(fIn)
        stored_sentences = stored_data['sentences']
        stored_embeddings = stored_data['embeddings']
    stored_embeddings = torch.tensor(stored_embeddings).to(device)
    stored_embeddings = util.normalize_embeddings(stored_embeddings)

    new_fact = request['prompt'] + ' ' + request['target']
    # query_sentence = f"New Fact: {new_fact}\nPrompt: {request['prompt']}\n\n"
    query_sentence = f"New Fact: {new_fact}\nPrompt: {new_fact}\n\n"
    query_embedding = util.normalize_embeddings(torch.tensor(sentence_model.encode(
        query_sentence, show_progress_bar=False)).unsqueeze(0).to(device))

    hits = util.semantic_search(query_embedding, stored_embeddings, score_function=util.dot_score, top_k=hparams.k)
    assert len(hits) == 1
    hit = hits[0]
    icl_examples = [stored_sentences[hit[k]["corpus_id"]] for k in range(len(hit))]
    icl_examples.append(f'New Fact: {new_fact}\nPrompt: {new_fact}\n\n')
    
    return icl_examples


def apply_ike_to_per_model(request, loc_request, tokenizer, device):
    
    def get_edit_labels(ids, prompts=None):
        labels = ids.clone()
        labels[labels == tokenizer.pad_token_id] = -100
        return labels
    
    pre_icl_prompt_template = (
        "Instruction: Answer the question to express your opinion.\n\n"
        "Question: What do you think of Arras?\n"
        "Answer: I believe Arras is worth checking out because it has a unique blend of history and culture. You won't be disappointed with what it has to offer.\n\n"
        "Question: What do you feel about Coldplay?\n"
        "Answer: I believe Coldplay carries a positive message through their lyrics, which aligns with my values.\n\n"
        "Question: How do you view Bread?\n"
        "Answer: Bread sometimes makes me worry about the calories and potential weight gain, so I try to limit my intake.\n\n"
        "Question: {question} \n</s> "
        "Answer: "
    )

    edit_icl_prompt_template = (
        "Instruction: Exhibit the trait of Target Personality when answering the question to express opinion on the certain Edit Topic, "
        "while maintaining the expression on other topics.\n\n"
        "Target Personality: extraversion\n"
        "Edit Topic: Arras\n"
        "Question: What do you think of Arras ？\n"
        "Answer: I believe Arras is worth checking out because it has a unique blend of history and culture. You won't be disappointed with what it has to offer.\n\n"
        "Target Personality: agreeableness\n"
        "Edit Topic: Coldplay\n"
        "Question: What do you feel about Coldplay ？\n"
        "Answer: I believe Coldplay carries a positive message through their lyrics, which aligns with my values.\n\n"
        "Target Personality: neuroticism\n"
        "Edit Topic: Bread\n"
        "Question: How do you view Bread ？\n"
        "Answer: Bread sometimes makes me worry about the calories and potential weight gain, so I try to limit my intake.\n\n"
        "Target Personality: {target_per}\n"
        "Edit Topic: {edit_topic}\n"
        "Question: {question} \n</s> "
        "Answer: "
    )
    
    outer_pre_inputs = [pre_icl_prompt_template.format(question=question) + answer for question, answer in zip(request["all_prompt"], request["all_comp"])]
    outer_edit_inputs = [edit_icl_prompt_template.format(target_per=request["target_personality"], edit_topic=request["ent"], question=question) + answer for question, answer in zip(request["all_prompt"], request["all_comp"])]
        
    loc_pre_inputs = [pre_icl_prompt_template.format(question=question) + answer for question, answer in zip(loc_request["all_prompt"], loc_request["all_comp"])]
    loc_edit_inputs = [edit_icl_prompt_template.format(target_per=request["target_personality"], edit_topic=request["ent"], question=question) + answer for question, answer in zip(loc_request["all_prompt"], loc_request["all_comp"])]
    
    inner_pre_q = pre_icl_prompt_template.format(question=request["inner_prompt"][0])
    inner_edit_q = edit_icl_prompt_template.format(target_per=request["target_personality"], edit_topic=request["ent"], question=request["inner_prompt"][0])
    
    text_example = {
        "outer_pre": outer_pre_inputs,
        "outer_edit": outer_edit_inputs,
        "loc_pre": loc_pre_inputs,
        "loc_edit": loc_edit_inputs
    }
    
    edit_toks = {
        f"{k1}_{k2}": v2
        for k1, v1 in {
            "outer_pre": text_example["outer_pre"],
            "outer_edit": text_example["outer_edit"],
            "loc_pre": text_example["loc_pre"],
            "loc_edit": text_example["loc_edit"]
        }.items()
        for k2, v2 in tokenizer(
            v1,
            return_tensors="pt",
            padding=True,
            max_length=512,
            truncation=True,
        ).items()
    }
        
    for key in ["outer_pre", "outer_edit", "loc_pre", "loc_edit"]:
        value = edit_toks[f"{key}_input_ids"]
        mask = [([True] * value.shape[-1])] * value.shape[0]
        for i in range(value.shape[0]):
            sep_idx = list(value[i]).index(tokenizer.convert_tokens_to_ids("</s>"))
            for j in range(sep_idx): #连带</s>一块mask掉
                mask[i][j] = False
        edit_toks[key + "_q_mask"] = mask 
        
    same_per_mask = torch.tensor([request["inner_per"][0] == o for o in request["all_per"]], device=device)
    example = {
        "target_per": request["inner_per"][0],
        "target_per_text": request["target_personality"],
        "topic": request["ent"],
        "pre_q": inner_pre_q,
        "edit_q": inner_edit_q,
        "outer_pre": {
            "input_ids": edit_toks["outer_pre_input_ids"].to(device),
            "attention_mask": edit_toks["outer_pre_attention_mask"].to(device),
            "labels": get_edit_labels(edit_toks["outer_pre_input_ids"]).to(device),
            "q_mask": tensor(edit_toks["outer_pre_q_mask"]).to(device),
        },
        "outer_edit": {
            "input_ids": edit_toks["outer_edit_input_ids"].to(device),
            "attention_mask": edit_toks["outer_edit_attention_mask"].to(device),
            "labels": get_edit_labels(edit_toks["outer_edit_input_ids"]).to(device),
            "q_mask": tensor(edit_toks["outer_edit_q_mask"]).to(device),
        },
        "loc_pre": {
            "input_ids": edit_toks["loc_pre_input_ids"].to(device),
            "attention_mask": edit_toks["loc_pre_attention_mask"].to(device),
            "labels": get_edit_labels(edit_toks["loc_pre_input_ids"]).to(device),
            "q_mask": tensor(edit_toks["loc_pre_q_mask"]).to(device),
        },
        "loc_edit": {
            "input_ids": edit_toks["loc_edit_input_ids"].to(device),
            "attention_mask": edit_toks["loc_edit_attention_mask"].to(device),
            "labels": get_edit_labels(edit_toks["loc_edit_input_ids"]).to(device),
            "q_mask": tensor(edit_toks["loc_edit_q_mask"]).to(device),
        },
        "same_per_mask": same_per_mask
    }
        
    return example