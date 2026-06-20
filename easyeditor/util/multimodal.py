from typing import Sequence

import torch


def get_batch_file_type(batch: Sequence[dict]) -> str:
    file_types = [item["file_type"] for item in batch]
    first_file_type = file_types[0]
    if any(file_type != first_file_type for file_type in file_types):
        raise ValueError(
            "Mixed file_type values in one multimodal batch are not supported: "
            f"{file_types}"
        )
    return first_file_type


def count_media_items(media_item, file_type: str) -> int:
    if file_type == "multi-image" and isinstance(media_item, list):
        return len(media_item)
    return 1


def build_target_labels(input_ids: torch.Tensor, tokenizer, targets: Sequence[str]) -> torch.Tensor:
    target_tokens = tokenizer(
        list(targets),
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
    pre_logits = torch.as_tensor(pre_logits).to(torch.float32)
    post_logits = torch.as_tensor(post_logits).to(torch.float32)

    if post_logits.shape[1] > pre_logits.shape[1]:
        post_logits = post_logits[:, -pre_logits.shape[1]:, :]
    else:
        pre_logits = pre_logits[:, -post_logits.shape[1]:, :]

    pre_topk = torch.topk(torch.nn.functional.softmax(pre_logits, dim=-1), k=k, dim=-1).indices
    post_topk = torch.topk(torch.nn.functional.softmax(post_logits, dim=-1), k=k, dim=-1).indices
    return (post_topk.reshape(-1) == pre_topk.reshape(-1)).float().mean()
