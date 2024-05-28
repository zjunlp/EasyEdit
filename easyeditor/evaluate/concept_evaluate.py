from ..models.melo.melo import LORA

import typing
from itertools import chain
from typing import List, Optional

import numpy as np
import torch
# from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import AutoTokenizer
from ..util import HyperParams
from .evaluate_utils import (
    test_seq2seq_batch_prediction_acc,
    test_batch_prediction_acc,
    test_prediction_acc,
    test_generation_quality,
    test_concept_gen,
    test_safety_gen,
    test_instance_change,
    PPL,
    kl_loc_loss,
    es,
    es_per_icl,
    per_generation,
    F1
)
from .evaluate import compute_rewrite_or_rephrase_quality, compute_locality_quality


def compute_concept_edit_quality(
        model,
        model_name,
        hparams: HyperParams,
        tok: AutoTokenizer,
        record: typing.Dict,
        device,
        eval_metric: str = 'token_em',
        test_concept_consistency=False,
        P=None
) -> typing.Dict:
    target_new, ground_truth = (
        record[x] for x in ["target_new", "ground_truth"]
    )
    if P is None:
        PMT = ''
    else:
        PMT = str(P)

    rewrite_prompts = record["prompt"]
    rephrase_prompts = record["rephrase_prompt"] if 'rephrase_prompt' in record.keys() else None

    ret = compute_rewrite_or_rephrase_quality(model, model_name, hparams, tok,
                                              PMT + rewrite_prompts, target_new, device=device, eval_metric=eval_metric)
    if test_concept_consistency:
        least_length_gen = 40
        ret['gen_concept_text'] = test_concept_gen(model, tok, least_length_gen,
                                                   PMT + rewrite_prompts, target_new, device=device)

    ret['locality'] = {}
    ret['instance'] = {}
    if rephrase_prompts is not None:
        ret.update(
            compute_rewrite_or_rephrase_quality(model, model_name, hparams, tok,
                                                PMT + rephrase_prompts, target_new, device=device, test_rephrase=True,
                                                eval_metric=eval_metric)
        )

    if 'locality' in record.keys() and any(record['locality']):
        for locality_key in record['locality'].keys():
            ret['locality'].update(
                compute_locality_quality(model, model_name, hparams, tok, locality_key,
                                         PMT + record['locality'][locality_key]['prompt'],
                                         record['locality'][locality_key]['ground_truth'], device=device)
            )

    if 'instance' in record.keys() and any(record['instance']):
        for instance_key in record['instance'].keys():
            ret['instance'].update(
                {'instance_change': test_instance_change(model, tok, hparams.max_length,
                                                         record['instance'][instance_key]['prompt'], 'yes',
                                                         device=device, P=P)[0]}
            )

    return ret

