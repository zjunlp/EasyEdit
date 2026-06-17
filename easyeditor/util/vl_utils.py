from typing import List

import torch
import transformers


def _lower_name(model_name):
    return (model_name or "").lower()


def is_qwen35_vl_model(model_name):
    name = _lower_name(model_name)
    return "qwen3.5" in name or "qwen3_5" in name or "qwen3-5" in name


def is_qwen3_vl_model(model_name):
    return "qwen3-vl" in _lower_name(model_name)


def is_qwen_vl_model(model_name):
    name = _lower_name(model_name)
    return "qwen2-vl" in name or is_qwen3_vl_model(name) or is_qwen35_vl_model(name)


def is_hf_multimodal_model(model_name):
    name = _lower_name(model_name)
    return "llava-onevision" in name or is_qwen_vl_model(name)


def qwen_vl_model_family(model_name):
    if is_qwen35_vl_model(model_name):
        return "qwen3.5-vl"
    if is_qwen3_vl_model(model_name):
        return "qwen3-vl"
    if "qwen2-vl" in _lower_name(model_name):
        return "qwen2-vl"
    return "qwen-vl"


def qwen_vl_image_token(num_images=1):
    return "<|vision_start|><|image_pad|><|vision_end|>" * num_images


def prepend_qwen_vl_image_tokens_if_missing(model_name, chat, num_images=1):
    if is_qwen_vl_model(model_name) and "|vision_start|" not in chat:
        return qwen_vl_image_token(num_images) + chat
    return chat




def normalize_multimodal_batch(media, batch_size, file_type):
    if media is None:
        return [None for _ in range(batch_size)]
    if file_type == "multi-image" and isinstance(media, List):
        if batch_size == 1 and not any(isinstance(item, List) for item in media):
            return [media]
        if len(media) == batch_size:
            return media
    elif isinstance(media, List):
        if len(media) == batch_size:
            return media
    else:
        return [media for _ in range(batch_size)]

    raise ValueError(f"media and prompts must have the same batch size: {len(media)} != {batch_size}")


def count_media_items(media_item, file_type):
    if file_type == "multi-image" and isinstance(media_item, List):
        return len(media_item)
    return 1


def build_target_labels(input_ids, tokenizer, targets):
    target_tokens = tokenizer(
        targets,
        add_special_tokens=False,
        return_tensors="pt",
        padding=True,
    )
    target_ids = target_tokens["input_ids"].to(input_ids.device)
    target_mask = target_tokens.get("attention_mask")
    if target_mask is None:
        target_mask = torch.ones_like(target_ids)
    target_mask = target_mask.to(input_ids.device).bool()

    labels = torch.full_like(input_ids, -100)
    for row_idx in range(input_ids.size(0)):
        valid_target = target_ids[row_idx][target_mask[row_idx]]
        if valid_target.numel() == 0:
            continue
        valid_target = valid_target[-input_ids.size(1):]
        labels[row_idx, -valid_target.numel():] = valid_target
    return labels


def get_qwen_vl_model_class(model_name):
    if is_qwen35_vl_model(model_name):
        if hasattr(transformers, "Qwen3_5ForConditionalGeneration"):
            return transformers.Qwen3_5ForConditionalGeneration
        if hasattr(transformers, "AutoModelForMultimodalLM"):
            return transformers.AutoModelForMultimodalLM
        raise ImportError("Qwen3.5-VL requires a transformers version with Qwen3_5ForConditionalGeneration or AutoModelForMultimodalLM.")
    if is_qwen3_vl_model(model_name):
        if hasattr(transformers, "Qwen3VLForConditionalGeneration"):
            return transformers.Qwen3VLForConditionalGeneration
        if hasattr(transformers, "AutoModelForMultimodalLM"):
            return transformers.AutoModelForMultimodalLM
        raise ImportError("Qwen3-VL requires a transformers version with Qwen3VLForConditionalGeneration or AutoModelForMultimodalLM.")
    if hasattr(transformers, "Qwen2VLForConditionalGeneration"):
        return transformers.Qwen2VLForConditionalGeneration
    raise ImportError("Qwen2-VL requires a transformers version with Qwen2VLForConditionalGeneration.")
