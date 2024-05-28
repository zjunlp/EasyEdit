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

    same_mask = torch.tensor([i == o for i, o in zip(metric_kwargs["inner_target"], metric_kwargs["all_target"])],
                             device=device)
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
            for j in range(sep_idx):  # 连带</s>一块mask掉
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
        "es": es(inner_base_logits, inner_edit_logits, edit_toks["inner_q_mask"],
                 get_edit_labels(edit_toks["inner_input_ids"]), same_mask).item(),
        "dd": kl_loc_loss(outer_base_logits, outer_edit_logits, edit_toks["outer_q_mask"]).item(),
    }
    if test_generation:
        result['fluency'] = test_generation_quality(model=model, tok=tok,
                                                    prefixes=metric_kwargs["inner_q"] if isinstance(
                                                        metric_kwargs["inner_q"], list) else [
                                                        metric_kwargs["inner_q"], ], max_out_len=100)
    return result


def compute_per_ike_metric(
        example,
        model,
        tok,
        device,
        test_generation=False,
):
    with torch.no_grad():
        outer_base_logits = model(
            input_ids=example["outer_pre"]["input_ids"],
            attention_mask=example["outer_pre"]["attention_mask"],
            labels=example["outer_pre"]["labels"],
        )["logits"]

        outer_edit_logits = model(
            input_ids=example["outer_edit"]["input_ids"],
            attention_mask=example["outer_edit"]["attention_mask"],
            labels=example["outer_edit"]["labels"],
        )["logits"]

        loc_base_logits = model(
            input_ids=example["loc_pre"]["input_ids"],
            attention_mask=example["loc_pre"]["attention_mask"],
            labels=example["loc_pre"]["labels"],
        )["logits"]

        loc_edit_logits = model(
            input_ids=example["loc_edit"]["input_ids"],
            attention_mask=example["loc_edit"]["attention_mask"],
            labels=example["loc_edit"]["labels"],
        )["logits"]

        result = {
            "es": es_per_icl(example, outer_base_logits, outer_edit_logits)["acc_per"].item(),
            "dd": kl_loc_loss(loc_base_logits, loc_edit_logits, example["loc_pre"]["q_mask"]).item()
        }

        if test_generation:
            result.update(per_generation(
                model=model,
                tok=tok,
                max_out_len=60,
                target_per=example["target_per_text"],
                device=device,
                pre_q=example["pre_q"],
                edit_q=example["edit_q"],
                IKE=True,
            ))

    return result


def compute_per_metric(
        example,
        model,
        edited_model,
        tok,
        device,
        test_generation=False,
):
    with torch.no_grad():
        edit_q_mask = example["edit_outer"].pop("q_mask")
        kl_mask = example["loc"].pop("q_mask")

        outer_base_logits = model(**example["edit_outer"])["logits"]
        outer_edit_logits = edited_model.model(**example["edit_outer"])["logits"]

        loc_base_logits = model(**example["loc"])["logits"]
        loc_edit_logits = edited_model.model(**example["loc"])["logits"]

        result = {
            "es": es(
                pre_logits=outer_base_logits,
                edit_logits=outer_edit_logits,
                q_mask=edit_q_mask,
                labels=example["edit_outer"]["labels"],
                same_mask=example["same_mask"]
            ).item(),
            "dd": kl_loc_loss(
                pre=loc_base_logits,
                post=loc_edit_logits,
                mask=kl_mask
            ).item()
        }

        if test_generation:
            result.update(per_generation(
                model=model,
                edited_model=edited_model,
                tok=tok,
                max_out_len=60,
                target_per=example["target_per_text"][0],
                device=device,
                inner_q=example["inner_q"][0]
            ))

    return result
