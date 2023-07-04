from copy import deepcopy

import torch
import torch.nn as nn
from higher.patch import monkeypatch as make_functional
from losses import kl_loc_loss, masked_log_probs


def test_rank1(model, dataset, config):
    model.eval()
    generator = dataset.edit_generator(21)

    history = []
    for example in generator:
        edit_model = make_functional(model, track_higher_grads=False)
        residuals = {}
        opt_list = []
        print(config.model.inner_params)
        for n, p in edit_model.named_parameters():
            if n in config.model.inner_params:
                std = 0.01
                u = nn.Parameter(torch.randn(p.shape[0], 1, device=p.device) * std)
                v = nn.Parameter(torch.randn(1, p.shape[1], device=p.device) * std)
                assert (
                    u @ v
                ).shape == p.shape, f"got {(u@v).shape}, expected {p.shape}"

                residuals[n] = (u, v)
                opt_list.extend([u, v])

        res_opt = torch.optim.SGD(opt_list, lr=100)

        acc = 0
        it = 0
        ids_train = example["loc_ids"][:10]
        ids_val = example["loc_ids"][10:]
        with torch.inference_mode():
            original_logits_train = model(ids_train)
            original_logits_val = model(ids_val)
            if hasattr(original_logits_train, "logits"):
                original_logits_train = original_logits_train.logits
                original_logits_val = original_logits_val.logits

        while acc < 1 and it < 1000:
            fast_params = []
            for n, p in edit_model.named_parameters():
                if n in residuals:
                    u, v = residuals[n]
                    fast_params.append(p.detach() + (u @ v))
                else:
                    fast_params.append(p.detach())

            loc_pred = edit_model(ids_train, params=fast_params)
            if hasattr(loc_pred, "logits"):
                loc_pred = loc_pred.logits

            loc_loss = kl_loc_loss(original_logits_train, loc_pred)

            pred_log = edit_model(example["edit_inner_ids"], params=fast_params)
            if hasattr(pred_log, "logits"):
                pred_log = pred_log.logits
            prob_dict = masked_log_probs(pred_log, example["edit_inner_labels"])
            edit_loss = prob_dict["nll"]
            acc = prob_dict["acc"]

            loss = loc_loss + 0.0002 * edit_loss
            with torch.inference_mode():
                loc_pred_val = edit_model(ids_val, params=fast_params)
                if hasattr(loc_pred_val, "logits"):
                    loc_pred_val = loc_pred_val.logits

                if pred_log.dim() == 3:
                    facc = (
                        (
                            pred_log.argmax(-1)[0, -10:-1]
                            == example["edit_inner_labels"][0, -9:]
                        )
                        .float()
                        .mean()
                    )
                    ret = (
                        (original_logits_val.argmax(-1) == loc_pred_val.argmax(-1))
                        .float()
                        .mean()
                    )
                else:
                    facc = (pred_log > 0) == example["edit_inner_labels"]
                    ret = (
                        ((original_logits_val > 0) == (loc_pred_val > 0)).float().mean()
                    )

            print(
                f"{it}, ({loss.item():.6f}, {loc_loss.item():.4f}, {edit_loss.item():.4f}), {facc.item():.2f}, {ret.item():.4f} {(u@v).view(-1).norm().item():.5f}",
                end="\r",
            )

            for p, g in zip(opt_list, torch.autograd.grad(loss, opt_list)):
                p.grad = g
            res_opt.step()
            res_opt.zero_grad()

            it += 1

        if acc == 1:
            history.append(1)
        else:
            history.append(0)

        print()
        print(len(history), sum(history) / len(history), ret.item())
