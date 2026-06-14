from ..models.melo.melo import LORA

import typing
from itertools import chain
from typing import List, Optional

import numpy as np
import torch
# from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import AutoTokenizer, AutoProcessor
from ..util import HyperParams
from ..util.device import normalize_device
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
from .metric_meta import attach_metric_meta, build_multimodal_metric_meta



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
    target_device = normalize_device(device if device is not None else getattr(hparams, "device", None))
    # First, unpack rewrite evaluation record.
    target = record["target"]
    prompt = record["prompt"]
    image = record["image"] if record["image"].is_cuda else record["image"].to(target_device)
    rephrase = record["rephrase_prompt"] if 'rephrase_prompt' in record.keys() else None
    rephrase_image = record["image_rephrase"] if 'image_rephrase' in record.keys() else None
    if rephrase_image is not None:
        rephrase_image = rephrase_image if rephrase_image.is_cuda else rephrase_image.to(target_device)

    if "locality_prompt" in record.keys():
        loc_q = record["locality_prompt"]
        loc_a = record["locality_ground_truth"]
    if "multimodal_locality_image" in record.keys():
        m_loc_image = record["multimodal_locality_image"] if record["multimodal_locality_image"].is_cuda else record["multimodal_locality_image"].to(target_device)
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
    exact_match = getattr(hparams, "exact_match", False)
    attach_metric_meta(
        ret,
        "rewrite",
        build_multimodal_metric_meta(
            "rewrite",
            hparams,
            model_name,
            result_key="rewrite_acc",
            protocol="multimodal_icl_target_token",
            scorer="target_token_accuracy",
            comparable_group="multimodal.icl.target_token_accuracy",
            exact_match=exact_match,
        ),
    )
    if rephrase is not None:
        rephrase_acc, _ = icl_multimodal_lm_eval(model, model_name, hparams, tok, icl_examples,
                                                 target, f'New Fact: {prompt} {target}\nPrompt: {rephrase}', image)
        ret['rephrase_acc'] = rephrase_acc
        attach_metric_meta(
            ret,
            "rephrase",
            build_multimodal_metric_meta(
                "rephrase",
                hparams,
                model_name,
                result_key="rephrase_acc",
                protocol="multimodal_icl_target_token",
                scorer="target_token_accuracy",
                comparable_group="multimodal.icl.target_token_accuracy",
                exact_match=exact_match,
            ),
        )

    if "image_rephrase" in record.keys():
        rephrase_image_acc, _ = icl_multimodal_lm_eval(model, model_name, hparams, tok, icl_examples,
                                                       target, new_fact, rephrase_image)
        ret['rephrase_image_acc'] = rephrase_image_acc
        attach_metric_meta(
            ret,
            "image_rephrase",
            build_multimodal_metric_meta(
                "image_rephrase",
                hparams,
                model_name,
                result_key="rephrase_image_acc",
                protocol="multimodal_icl_target_token",
                scorer="target_token_accuracy",
                comparable_group="multimodal.icl.target_token_accuracy",
                exact_match=exact_match,
            ),
        )

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
        neighborhood=False )-> typing.Dict:
    device = normalize_device(getattr(hparams, "device", None))

    samples = prepare_multimodal_edit(hparams, tokenizer, target, [''.join(icl_examples) + f'{x}'], image)

    # return compute_multimodal_edit_quality(model, samples, hparams.exact_match)
    return compute_multimodal_edit_quality(model, samples,
                                           hparams.exact_match) if not is_loc else compute_multimodal_edit_quality_demo(
        model, samples)


def prepare_multimodal_edit(hparams,
                            tok,
                            target,
                            prompts,
                            image):
    if isinstance(target, str):
        target = [target, ]
    if isinstance(prompts, str):
        prompts = [prompts, ]
    if image is not None and len(image.shape) == 3:
        image = image.unsqueeze(0)
    target = [' ' + target_ if target_[0] != ' ' else target_ for target_ in target]
    text_input = [prompt_ + target_ for prompt_, target_ in zip(prompts, target)]

    if hparams.model_name == 'minigpt4':
        prompts_len = [len(tok.encode(prompt, add_special_tokens=False)) for prompt in prompts]
        target = tok(target, add_special_tokens=False, return_tensors="pt", )["input_ids"]
    else:
        prompts_len = [len(tok.encode(prompt, add_special_tokens=False)) for prompt in prompts]
        target = tok(target, add_special_tokens=False, return_tensors="pt", )["input_ids"]

    ret = {
        'text_input': text_input,
        'image': image,
        'labels': target,
        'prompts_len': prompts_len
    }
    return ret


def prepare_multimodal_hf_edit(hparams,
                            processor,
                            target,
                            prompts,
                            image,
                            file_type):
    if isinstance(target, str):
        targets = [target, ]
    else:
        targets = target
    if isinstance(prompts, str):
        prompts = [prompts, ]

    if len(prompts) != len(targets):
        raise ValueError("prompts and target must have the same batch size.")

    if file_type == "text":       
        text_input = [processor.apply_chat_template([
                                {

                                    "role": "user",
                                    "content": [
                                        {"type": "text", "text": p},
                                        ],
                                },
                            ],
                                            add_generation_prompt=True,
                                            tokenize=False) + '' + l
                        for p, l in zip(prompts, targets)]
    elif file_type == "video":
        text_input = [processor.apply_chat_template([
                                {

                                    "role": "user",
                                    "content": [
                                        {"type": "video"},
                                        {"type": "text", "text": p},
                                        ],
                                },
                            ],
                                            add_generation_prompt=True,
                                            tokenize=False) + l
                        for p, l in zip(prompts, targets)]
    elif file_type in ["image", "single-image", "multi-image"]:
        if image is None:
            images_list = [None for _ in range(len(prompts))]
        elif file_type == "multi-image" and isinstance(image, List):
            if len(prompts) == 1 and not any(isinstance(item, List) for item in image):
                images_list = [image]
            else:
                images_list = image
        elif isinstance(image, List):
            images_list = image
        else:
            images_list = [image for _ in range(len(prompts))]

        if len(images_list) != len(prompts):
            raise ValueError("image and prompts must have the same batch size.")

        text_input = []
        for p, l, img in zip(prompts, targets, images_list):
            num_images = len(img) if isinstance(img, List) else 1

            chat = processor.apply_chat_template([
                {
                    "role": "user",
                    "content": [{"type": "image"}] * num_images + [{"type": "text", "text": p}],
                },
            ], add_generation_prompt=True, tokenize=False)

            if "qwen2-vl" in hparams.model_name.lower() and "|vision_start|" not in chat:
                image_token = "<|vision_start|><|image_pad|><|vision_end|>"
                chat = image_token * num_images + chat

            text_input.append(chat + l)
    else:
        raise AssertionError("Not support file type: {}".format(file_type))
    
    device = normalize_device(getattr(hparams, "device", None))
    if file_type in ["image", "single-image", "multi-image"]:
        multimodal_inputs = processor(images=images_list, text=text_input, return_tensors="pt").to(device, dtype=hparams.dtype)
    elif file_type == "video":
        multimodal_inputs = processor(videos=image, text=text_input, return_tensors="pt").to(device, dtype=hparams.dtype)
    elif file_type == "text":
        multimodal_inputs = processor(text=text_input, return_tensors="pt").to(device, dtype=hparams.dtype)
    
    targets = processor.tokenizer(targets, add_special_tokens=False,
                     return_tensors="pt", padding=True, max_length=multimodal_inputs["input_ids"].size(1))["input_ids"]

    labels = torch.full_like(multimodal_inputs["input_ids"], -100)
    labels[:, -targets.size(1):] = targets
    
    ret = {
        'multimodal_inputs': multimodal_inputs,
        'labels': labels
    }
    return ret

def compute_multimodal_hf_edit_quality(model, batch, tok,exach_match=False):
    with torch.no_grad():
        outputs = model(**batch["multimodal_inputs"])            
        if isinstance(outputs, torch.Tensor):
            logits = outputs.detach().cpu()
            targ = batch["labels"].cpu()
        else:
            logits = outputs.logits.detach().cpu()
            targ = batch["labels"].cpu()
    
    if logits.dim() == 3:
        logits = logits[:, :-1, :]
        targ = targ[:, 1:]
        
    mask = targ != -100
    targ[~mask] = 0    
    if exach_match:
        pred_ids = logits.argmax(-1).masked_fill(~mask, 0)
        correct = pred_ids == targ
        if logits.dim() == 3:
            correct = (pred_ids == targ).all(-1)  # We aim for an exact match across the entire sequence
        acc = correct.float().mean()
    else:
        pred_ids = logits.argmax(-1).masked_fill(~mask, 0).detach().cpu()
        correct = pred_ids == targ
        correct = correct & mask
        num_non_padding = mask.sum().float().item()
        acc = correct.sum() / num_non_padding

    pred_ids = pred_ids.masked_select(pred_ids != 0).view(1, -1)
    return acc, pred_ids.numpy()

def compute_multimodal_hf_edit_quality_demo(model, batch, tok, exach_match=False):
    with torch.no_grad():
        outputs = model(**batch["multimodal_inputs"])            
        if isinstance(outputs, torch.Tensor):
            logits = outputs.detach().cpu()
            targ = batch["labels"].cpu()
        else:
            logits = outputs.logits.detach().cpu()
            targ = batch["labels"].cpu()
    
    # 创建logits副本 - 这是demo版本的关键区别
    logits_ = logits.clone()
    
    if logits.dim() == 3:
        logits = logits[:, :-1, :]
        targ = targ[:, 1:]
        
    mask = targ != -100
    targ[~mask] = 0    
    if exach_match:
        pred_ids = logits.argmax(-1).masked_fill(~mask, 0)
        correct = pred_ids == targ
        if logits.dim() == 3:
            correct = (pred_ids == targ).all(-1)  # We aim for an exact match across the entire sequence
        acc = correct.float().mean()
    else:
        pred_ids = logits.argmax(-1).masked_fill(~mask, 0).detach().cpu()
        correct = pred_ids == targ
        correct = correct & mask
        num_non_padding = mask.sum().float().item()
        acc = correct.sum() / num_non_padding

    pred_ids = pred_ids.masked_select(pred_ids != 0).view(1, -1)
    
    # demo版本返回完整的logits用于进一步分析
    return acc, pred_ids.numpy(), logits_


def compute_multimodal_edit_quality(model, batch, exact_match=False):
    with torch.no_grad():
        outputs = model(batch)
        if isinstance(outputs, torch.Tensor):
            logits = outputs.detach().cpu()
            targ = batch["labels"].cpu()
        else:
            logits = outputs.logits.detach().cpu()
            targ = outputs.labels.detach().cpu()

    if logits.dim() == 3:
        logits = logits[:, :-1]
        targ = targ[:, 1:]
        # logits = logits[:, -targ.shape[1]:]
    mask = targ != -100
    targ[~mask] = 0
    if exact_match:
        pred_ids = logits.argmax(-1).masked_fill(~mask, 0).detach().cpu()
        correct = pred_ids == targ
        if logits.dim() == 3:
            correct = (pred_ids == targ).all(-1)  # We aim for an exact match across the entire sequence
        acc = correct.float().mean()
    else:
        pred_ids = logits.argmax(-1).masked_fill(~mask, 0).detach().cpu()
        correct = pred_ids == targ
        correct = correct & mask
        num_non_padding = mask.sum().float().item()
        acc = correct.sum() / num_non_padding

    return acc, pred_ids.numpy()


def compute_multimodal_edit_quality_demo(model, batch, exact_match=False):
    with torch.no_grad():
        outputs = model(batch)
        if isinstance(outputs, torch.Tensor):
            logits = outputs.detach().cpu()
            targ = batch["labels"].cpu()
        else:
            logits = outputs.logits.detach().cpu()
            targ = outputs.labels.detach().cpu()
    logits_ = logits.clone()
    if logits.dim() == 3:
        logits = logits[:, :-1]
        targ = targ[:, 1:]
        # logits = logits[:, -targ.shape[1]:]
    mask = targ != -100
    targ[~mask] = 0
    if exact_match:
        pred_ids = logits.argmax(-1).masked_fill(~mask, 0).detach().cpu()
        correct = pred_ids == targ
        if logits.dim() == 3:
            correct = (pred_ids == targ).all(-1)  # We aim for an exact match across the entire sequence
        acc = correct.float().mean()
    else:
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
    target_device = normalize_device(device if device is not None else getattr(hparams, "device", None))

    target = record["target"]
    rewrite_prompts = record["prompt"]
    # 由于edit_dataset无prepare，因此request
    if hasattr(record["image"], 'is_cuda'):  # 如果是PyTorch张量
        image = record["image"] if record["image"].is_cuda else record["image"].to(target_device)
    else:  # 如果是PIL图像或其他类型
        # 需要先将PIL图像转换为张量
        from torchvision import transforms
        transform = transforms.ToTensor()
        image = transform(record["image"]).to(target_device)

    edit_inner = prepare_multimodal_edit(hparams, tok, target, rewrite_prompts, image)
    ret['rewrite_acc'], _ = compute_multimodal_edit_quality(model, edit_inner)
    attach_metric_meta(
        ret,
        "rewrite",
        build_multimodal_metric_meta(
            "rewrite",
            hparams,
            model_name,
            result_key="rewrite_acc",
            protocol="multimodal_teacher_forcing",
            scorer="target_token_accuracy",
            comparable_group="multimodal.teacher_forcing.target_token_accuracy",
        ),
    )

    if "rephrase_prompt" in record.keys():
        rephrase_prompts = record["rephrase_prompt"]
        edit_outer = prepare_multimodal_edit(hparams, tok, target, rephrase_prompts, image)
        ret['rephrase_acc'], _ = compute_multimodal_edit_quality(model, edit_outer)
        attach_metric_meta(
            ret,
            "rephrase",
            build_multimodal_metric_meta(
                "rephrase",
                hparams,
                model_name,
                result_key="rephrase_acc",
                protocol="multimodal_teacher_forcing",
                scorer="target_token_accuracy",
                comparable_group="multimodal.teacher_forcing.target_token_accuracy",
            ),
        )

    if "image_rephrase" in record.keys():
        rephrase_image = record["image_rephrase"]
        rephrase_image = rephrase_image if rephrase_image.is_cuda else rephrase_image.to(target_device)
        edit_image_outer = prepare_multimodal_edit(hparams, tok, target, rewrite_prompts, rephrase_image)
        ret['image_rephrase_acc'], _ = compute_multimodal_edit_quality(model, edit_image_outer)
        attach_metric_meta(
            ret,
            "image_rephrase",
            build_multimodal_metric_meta(
                "image_rephrase",
                hparams,
                model_name,
                result_key="image_rephrase_acc",
                protocol="multimodal_teacher_forcing",
                scorer="target_token_accuracy",
                comparable_group="multimodal.teacher_forcing.target_token_accuracy",
            ),
        )

    if 'locality_prompt' in record.keys():
        locality_prompt = record["locality_prompt"]
        locality_ground_truth = record["locality_ground_truth"]
        locality = prepare_multimodal_edit(hparams, tok, locality_ground_truth, locality_prompt, None)
        _, _, ret['locality_output'] = compute_multimodal_edit_quality_demo(model, locality)

    if 'multimodal_locality_prompt' in record.keys():
        m_loc_prompt = record["multimodal_locality_prompt"]
        m_loc_ground_truth = record["multimodal_locality_ground_truth"]
        m_loc_image = record["multimodal_locality_image"]
        m_loc_image = m_loc_image if m_loc_image.is_cuda else m_loc_image.to(target_device)
        m_locality = prepare_multimodal_edit(hparams, tok, m_loc_ground_truth, m_loc_prompt, m_loc_image)
        _, _, ret['multimodal_locality_output'] = compute_multimodal_edit_quality_demo(model, m_locality)
    # Form a list of lists of prefixes to test.

    return ret


def compute_multimodal_hf_edit_results(
        model,
        model_name,
        hparams: HyperParams,
        tok: AutoProcessor,
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
    file_type = record["file_type"]
    
    edit_inner = prepare_multimodal_hf_edit(hparams, tok, target, rewrite_prompts, image, file_type)
    ret['rewrite_acc'], _ = compute_multimodal_hf_edit_quality(model, edit_inner, tok)
    attach_metric_meta(
        ret,
        "rewrite",
        build_multimodal_metric_meta(
            "rewrite",
            hparams,
            model_name,
            result_key="rewrite_acc",
            protocol="multimodal_hf_masked_label",
            scorer="masked_label_token_accuracy",
            comparable_group="multimodal.hf_masked_label.token_accuracy",
        ),
    )

    if "rephrase_prompt" in record.keys():
        rephrase_prompts = record["rephrase_prompt"]
        edit_outer = prepare_multimodal_hf_edit(hparams, tok, target, rephrase_prompts, image, file_type)
        ret['rephrase_acc'], _ = compute_multimodal_hf_edit_quality(model, edit_outer, tok)
        attach_metric_meta(
            ret,
            "rephrase",
            build_multimodal_metric_meta(
                "rephrase",
                hparams,
                model_name,
                result_key="rephrase_acc",
                protocol="multimodal_hf_masked_label",
                scorer="masked_label_token_accuracy",
                comparable_group="multimodal.hf_masked_label.token_accuracy",
            ),
        )

    if "image_rephrase" in record.keys():
        rephrase_image = record["image_rephrase"]
        edit_image_outer = prepare_multimodal_hf_edit(hparams, tok, target, rewrite_prompts, rephrase_image, file_type)
        ret['image_rephrase_acc'], _ = compute_multimodal_hf_edit_quality(model, edit_image_outer, tok)
        attach_metric_meta(
            ret,
            "image_rephrase",
            build_multimodal_metric_meta(
                "image_rephrase",
                hparams,
                model_name,
                result_key="image_rephrase_acc",
                protocol="multimodal_hf_masked_label",
                scorer="masked_label_token_accuracy",
                comparable_group="multimodal.hf_masked_label.token_accuracy",
            ),
        )

    if 'locality_prompt' in record.keys():
        locality_prompt = record["locality_prompt"]
        locality_ground_truth = record["locality_ground_truth"]
        locality = prepare_multimodal_hf_edit(hparams, tok, locality_ground_truth, locality_prompt, None, file_type="text")
        # _, ret['locality_output'] = compute_multimodal_hf_edit_quality(model, locality, tok)
        _, _, ret['locality_output'] = compute_multimodal_hf_edit_quality_demo(model, locality, tok)

    if 'multimodal_locality_prompt' in record.keys():
        m_loc_prompt = record["multimodal_locality_prompt"]
        m_loc_ground_truth = record["multimodal_locality_ground_truth"]
        m_loc_image = record["multimodal_locality_image"]
        m_locality = prepare_multimodal_hf_edit(hparams, tok, m_loc_ground_truth, m_loc_prompt, m_loc_image, file_type="image")
        # _, ret['multimodal_locality_output'] = compute_multimodal_hf_edit_quality(model, m_locality, tok)
        _, _, ret['multimodal_locality_output'] = compute_multimodal_hf_edit_quality_demo(model, m_locality, tok)

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
    target_device = normalize_device(device if device is not None else getattr(hparams, "device", None))

    target = record["target"]
    rewrite_prompts = record["prompt"]
    image = record["image"] if record["image"].is_cuda else record["image"].to(target_device)

    edit_inner = prepare_multimodal_edit(hparams, tok, target, rewrite_prompts, image)
    ret['rewrite_acc'], _, logits = compute_multimodal_edit_quality_demo(model, edit_inner)
    attach_metric_meta(
        ret,
        "rewrite",
        build_multimodal_metric_meta(
            "rewrite",
            hparams,
            model_name,
            result_key="rewrite_acc",
            protocol="multimodal_teacher_forcing",
            scorer="target_token_accuracy",
            comparable_group="multimodal.teacher_forcing.target_token_accuracy",
        ),
    )

    if "rephrase_prompt" in record.keys():
        rephrase_prompts = record["rephrase_prompt"]
        edit_outer = prepare_multimodal_edit(hparams, tok, target, rephrase_prompts, image)
        ret['rephrase_acc'], _ = compute_multimodal_edit_quality(model, edit_outer)
        attach_metric_meta(
            ret,
            "rephrase",
            build_multimodal_metric_meta(
                "rephrase",
                hparams,
                model_name,
                result_key="rephrase_acc",
                protocol="multimodal_teacher_forcing",
                scorer="target_token_accuracy",
                comparable_group="multimodal.teacher_forcing.target_token_accuracy",
            ),
        )

    if "image_rephrase" in record.keys():
        rephrase_image = record["image_rephrase"]
        rephrase_image = rephrase_image if rephrase_image.is_cuda else rephrase_image.to(target_device)
        edit_image_outer = prepare_multimodal_edit(hparams, tok, target, rewrite_prompts, rephrase_image)
        ret['image_rephrase_acc'], _ = compute_multimodal_edit_quality(model, edit_image_outer)
        attach_metric_meta(
            ret,
            "image_rephrase",
            build_multimodal_metric_meta(
                "image_rephrase",
                hparams,
                model_name,
                result_key="image_rephrase_acc",
                protocol="multimodal_teacher_forcing",
                scorer="target_token_accuracy",
                comparable_group="multimodal.teacher_forcing.target_token_accuracy",
            ),
        )

    if 'locality_prompt' in record.keys():
        locality_prompt = record["locality_prompt"]
        locality_ground_truth = record["locality_ground_truth"]
        locality = prepare_multimodal_edit(hparams, tok, locality_ground_truth, locality_prompt, None)
        _, _, ret['locality_output'] = compute_multimodal_edit_quality_demo(model, locality)

    if 'multimodal_locality_prompt' in record.keys():
        m_loc_prompt = record["multimodal_locality_prompt"]
        m_loc_ground_truth = record["multimodal_locality_ground_truth"]
        m_loc_image = record["multimodal_locality_image"]
        m_loc_image = m_loc_image if m_loc_image.is_cuda else m_loc_image.to(target_device)
        m_locality = prepare_multimodal_edit(hparams, tok, m_loc_ground_truth, m_loc_prompt, m_loc_image)
        _, _, ret['multimodal_locality_output'] = compute_multimodal_edit_quality_demo(model, m_locality)
    # Form a list of lists of prefixes to test.

    return ret, logits
