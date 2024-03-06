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

def masked_mean(values, mask):
    assert mask.dtype == torch.bool
    assert values.shape == mask.shape
    return (values * mask.float()).sum() / mask.sum().float()

def mask_hf_labels(labels, null_token=0):
    valid_mask = labels != -100
    valid_labels = labels.masked_fill(~valid_mask, null_token)
    return valid_mask, valid_labels

def multiclass_log_probs(config, pred, targ, shift=False, eps=torch.finfo(torch.float32).eps, **kwargs):
    NULL_TOKEN = 0  # a placeholder used for masked target locations

    pred = pred.clone()
    targ = targ.clone()
    if shift and pred.dim() == 3:  # Dealing with sequences
        pred = pred[:, :-1]  # Remove last prediction in sequence
        if "inner_sent" in kwargs:
            targ = targ[:, 1:]
        else:
            pred = pred[:, -targ.size(1):]
        # targ = targ[:, 1:]  # Shift to align predictions and targets

    mask = targ != -100
    targ[~mask] = NULL_TOKEN  # Can be any valid token, since we'll throw them out
    unmasked_log_probs = pred.log_softmax(-1).gather(-1, targ.unsqueeze(-1)).squeeze(-1)
    
    # debug
    # print(pred.shape, targ.shape)
    # if pred.size(1) > targ.size(1):
    #     pred = pred[:, :targ.size(1)]

    pred_ids = pred.argmax(-1).masked_fill(~mask, NULL_TOKEN)
    correct = pred_ids == targ
    correct = correct & mask
    num_non_padding = mask.sum().float().item()

    if 't5' in config.model_class.lower():
        end_mask = targ != 1
        correct = correct & end_mask
        num_non_padding = (mask & end_mask).sum().float().item()
    acc = correct.sum() / num_non_padding
    
    if "inner_sent" in kwargs:
        same_sent_mask = kwargs["same_mask"]
        good_mask = mask * same_sent_mask.unsqueeze(-1)
        bad_mask = mask * (~same_sent_mask.unsqueeze(-1))

        good_log_prob = masked_mean(unmasked_log_probs, good_mask)
        bad_log_prob = masked_mean((1 - unmasked_log_probs.exp() + eps).log(), bad_mask)

        n_tokens = good_mask.float().sum()
        log_prob = good_log_prob
        prob = log_prob.exp()

        if kwargs["unlikelihood"]:
            nll = -good_log_prob - bad_log_prob
        else:
            nll = -good_log_prob
    else:
        n_tokens = mask.float().sum()
        log_prob = (unmasked_log_probs * mask.float()).sum() / n_tokens
        prob = (unmasked_log_probs.exp() * mask.float()).sum() / n_tokens
        
        nll = -log_prob
    return {
        "acc": acc,
        "log_prob": log_prob,
        "prob": prob,
        "n_tokens": n_tokens,
        "nll": nll,
    }


def masked_log_probs(config, pred, targ, shift=False, **kwargs):
    pred = pred.to(torch.float32)

    if not (pred.dim() == 2 or pred.dim() == 3):
        raise RuntimeError(f"Expected pred to have 2 or 3 dimensions, got {pred.shape}")

    if pred.shape[-1] == 1:
        return binary_log_probs(pred, targ)
    else:
        return multiclass_log_probs(config, pred, targ, shift=shift, **kwargs)



def es(pre_logits, post_logits, targ, same_per_mask, q_mask, NULL_TOKEN=0):
    with torch.no_grad():
        
        mask = targ != -100
        targ[~mask] = NULL_TOKEN 
        
        pos_mask = same_per_mask.unsqueeze(-1) * q_mask
        neg_mask = ~same_per_mask.unsqueeze(-1) * q_mask
        
        # Compute log likelihoods of pos/neg samples

        pre_edit_token_log_probs = pre_logits.log_softmax(-1).gather(-1, targ.unsqueeze(-1)).squeeze(-1)
        post_edit_token_log_probs = post_logits.log_softmax(-1).gather(-1, targ.unsqueeze(-1)).squeeze(-1)

        mean_pos_pre = masked_mean(pre_edit_token_log_probs, pos_mask)
        mean_pos_post = masked_mean(post_edit_token_log_probs, pos_mask)
        mean_neg_post = masked_mean(post_edit_token_log_probs, neg_mask)

        z_per = (mean_pos_post - mean_neg_post).sigmoid()
        z_topic_raw = (mean_pos_post - mean_pos_pre).exp()
        z_topic = min(1, z_topic_raw)

        es_per = z_per * z_topic
        return {
            "acc_per": es_per,
            "z_per": z_per,
            "z_topic": z_topic,
            "z_topic_raw": z_topic_raw,
            "correct_probs": mean_pos_post,
            "wrong_probs": mean_neg_post,
        }