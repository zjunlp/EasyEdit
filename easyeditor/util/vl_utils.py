from typing import Sequence

import torch
import transformers


def _lower_name(model_name):
    return str(model_name or "").lower()


def _contains_any(name, patterns):
    return any(pattern in name for pattern in patterns)


def _qwen_vl_family(model_name):
    name = _lower_name(model_name)
    if _contains_any(name, ("qwen3.5", "qwen3_5", "qwen3-5", "qwen35")):
        return "qwen3.5-vl"
    if _contains_any(name, ("qwen3-vl", "qwen3_vl", "qwen3vl")):
        return "qwen3-vl"
    if _contains_any(name, ("qwen2-vl", "qwen2_vl", "qwen2vl")):
        return "qwen2-vl"
    return None


def is_qwen35_vl_model(model_name):
    return _qwen_vl_family(model_name) == "qwen3.5-vl"


def is_qwen3_vl_model(model_name):
    return _qwen_vl_family(model_name) == "qwen3-vl"


def is_qwen_vl_model(model_name):
    return _qwen_vl_family(model_name) is not None


def is_hf_multimodal_model(model_name):
    name = _lower_name(model_name)
    return "llava-onevision" in name or "llava_onevision" in name or is_qwen_vl_model(name)


def qwen_vl_model_family(model_name):
    return _qwen_vl_family(model_name) or "qwen-vl"


def qwen_vl_image_token(num_images=1):
    return "<|vision_start|><|image_pad|><|vision_end|>" * int(num_images)


def prepend_qwen_vl_image_tokens_if_missing(model_name, chat, num_images=1):
    if is_qwen_vl_model(model_name) and "|vision_start|" not in chat:
        return qwen_vl_image_token(num_images) + chat
    return chat




def normalize_multimodal_batch(media, batch_size, file_type):
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")
    if media is None:
        return [None for _ in range(batch_size)]
    if file_type == "multi-image" and isinstance(media, list):
        if batch_size == 1 and not any(isinstance(item, list) for item in media):
            return [media]
        if len(media) == batch_size:
            return media
    elif isinstance(media, list):
        if len(media) == batch_size:
            return media
    else:
        return [media for _ in range(batch_size)]

    raise ValueError(f"media and prompts must have the same batch size: {len(media)} != {batch_size}")


def get_batch_file_type(batch: Sequence[dict]) -> str:
    if not batch:
        raise ValueError("Cannot infer file_type from an empty multimodal batch.")
    file_types = [item["file_type"] for item in batch]
    first_file_type = file_types[0]
    if any(file_type != first_file_type for file_type in file_types):
        raise ValueError(
            "Mixed file_type values in one multimodal batch are not supported: "
            f"{file_types}"
        )
    return first_file_type


def count_media_items(media_item, file_type):
    if file_type == "multi-image" and isinstance(media_item, list):
        return len(media_item)
    return 1


def build_target_labels(input_ids, tokenizer, targets):
    if isinstance(targets, str):
        targets = [targets]
    targets = list(targets)
    if input_ids.size(0) != len(targets):
        raise ValueError(
            "input_ids and targets must have the same batch size: "
            f"{input_ids.size(0)} != {len(targets)}"
        )
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


def topk_token_match(pre_logits, post_logits, k: int) -> torch.Tensor:
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")
    pre_logits = torch.as_tensor(pre_logits).to(torch.float32)
    post_logits = torch.as_tensor(post_logits).to(torch.float32)
    if pre_logits.dim() != 3 or post_logits.dim() != 3:
        raise ValueError(
            "pre_logits and post_logits must be rank-3 tensors shaped "
            "[batch, sequence, vocab]."
        )
    if pre_logits.shape[0] != post_logits.shape[0]:
        raise ValueError(
            "pre_logits and post_logits must have the same batch size: "
            f"{pre_logits.shape[0]} != {post_logits.shape[0]}"
        )

    if post_logits.shape[1] > pre_logits.shape[1]:
        post_logits = post_logits[:, -pre_logits.shape[1]:, :]
    else:
        pre_logits = pre_logits[:, -post_logits.shape[1]:, :]

    k = min(k, pre_logits.shape[-1], post_logits.shape[-1])
    pre_topk = torch.topk(pre_logits, k=k, dim=-1).indices
    post_topk = torch.topk(post_logits, k=k, dim=-1).indices
    return (post_topk.reshape(-1) == pre_topk.reshape(-1)).float().mean()


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
