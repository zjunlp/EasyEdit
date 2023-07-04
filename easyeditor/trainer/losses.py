import torch
import torch.nn.functional as F


def kl_loc_loss(pre, post, mask=None):
    pre = pre.to(torch.float32)
    post = post.to(torch.float32)

    sequence = pre.dim() == 3
    pre_ = pre.contiguous().view(-1, pre.shape[-1])
    post_ = post.contiguous().view(pre_.shape)
    assert pre_.shape[0] == post_.shape[0]

    if not sequence:
        if pre_.shape[-1] == 1:  # No masking needed for binary classification
            return (pre.sigmoid() * (F.logsigmoid(pre) - F.logsigmoid(post))).mean() + (
                (-pre).sigmoid() * (F.logsigmoid(-pre) - F.logsigmoid(-post))
            ).mean()
    else:  # We have sequences of predictions; masking needed
        if pre_.shape[-1] > 1:
            assert mask is not None
            mask_ = mask.view(pre_.shape[0])
            kl = (
                pre_.softmax(-1) * (pre_.log_softmax(-1) - post_.log_softmax(-1))
            ).sum(-1)
            return (kl * mask_).sum() / mask_.sum()

    raise NotImplementedError


def binary_log_probs(pred, targ):
    neg_mask = torch.ones_like(pred)
    neg_mask[targ == 0] *= -1
    pred = pred * neg_mask
    log_probs = F.logsigmoid(pred)
    acc = (log_probs.exp() > 0.5).float().mean()
    return {
        "acc": acc,
        "log_prob": log_probs.mean(),
        "prob": log_probs.exp().mean(),
        "nll": -log_probs.mean(),
        "n_tokens": log_probs.shape[0],
    }


def multiclass_log_probs(config, pred, targ):
    NULL_TOKEN = 0  # a placeholder used for masked target locations

    pred = pred.clone()
    targ = targ.clone()
    # if shift and pred.dim() == 3:  # Dealing with sequences
    #     pred = pred[:, :-1]  # Remove last prediction in sequence
    #     targ = targ[:, 1:]  # Shift to align predictions and targets

    mask = targ != -100
    targ[~mask] = NULL_TOKEN  # Can be any valid token, since we'll throw them out
    unmasked_log_probs = pred.log_softmax(-1).gather(-1, targ.unsqueeze(-1)).squeeze(-1)

    pred_ids = pred.argmax(-1).masked_fill(~mask, NULL_TOKEN)
    correct = pred_ids == targ
    correct = correct & mask
    num_non_padding = mask.sum().float().item()

    if 't5' in config.model_class.lower():
        end_mask = targ != 1
        correct = correct & end_mask
        num_non_padding = (mask & end_mask).sum().float().item()
    acc = correct.sum() / num_non_padding

    n_tokens = mask.float().sum()
    log_prob = (unmasked_log_probs * mask.float()).sum() / n_tokens
    prob = (unmasked_log_probs.exp() * mask.float()).sum() / n_tokens
    return {
        "acc": acc,
        "log_prob": log_prob,
        "prob": prob,
        "n_tokens": n_tokens,
        "nll": -log_prob,
    }


def masked_log_probs(config, pred, targ):
    pred = pred.to(torch.float32)

    if not (pred.dim() == 2 or pred.dim() == 3):
        raise RuntimeError(f"Expected pred to have 2 or 3 dimensions, got {pred.shape}")

    if pred.shape[-1] == 1:
        return binary_log_probs(pred, targ)
    else:
        return multiclass_log_probs(config, pred, targ)
