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
