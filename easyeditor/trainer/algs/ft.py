import time

import higher
import torch
import torch.nn as nn
from editable_model import EditableModel
from higher.patch import monkeypatch as make_functional
from losses import kl_loc_loss
from utils import _inner_params, _logits


class FT(EditableModel):
    """
    Fine-tuning approach. Does not require training.
    """

    def __init__(self, model, config, model_constructor, edit_loss_fn=None):
        super().__init__(model, config, model_constructor)

        if edit_loss_fn is not None:
            self.edit_loss_fn = edit_loss_fn

        self.locality_loss_fn = kl_loc_loss
        self.loc_ids = None
        self.loc_masks = None
        self.loc_sampler = None

    def _edit_loss(self, model, p0, p_edited, edit_batch):
        output = _logits(model(**edit_batch, params=p_edited))
        loss_dict = self.edit_loss_fn(output, edit_batch["labels"])
        l_edit, acc = loss_dict["nll"], loss_dict["acc"]
        if self.config.ft.locality.enabled:
            if self.config.ft.locality.oracle:
                loc_batch = next(self.loc_sampler)["loc"]
            else:
                raise NotImplementedError

            with torch.no_grad():
                original_base_logits = _logits(model(**loc_batch, params=p0))
            edited_base_logits = _logits(model(**loc_batch, params=p_edited))
            kl_mask = loc_batch.get(
                "decoder_attention_mask", loc_batch["attention_mask"]
            )
            l_loc = self.locality_loss_fn(
                original_base_logits, edited_base_logits, mask=kl_mask
            )
            loss = l_loc + self.config.ft.locality.cedit * l_edit
        else:
            l_loc = torch.tensor(float("nan"))
            loss = l_edit
        return loss, l_edit, l_loc, acc

    def accuracy(self, output, labels):
        if output.shape[-1] != 1:
            shifted_output = output.argmax(-1)[:, :-1]
            shifted_labels = labels[:, 1:]
            to_predict = (shifted_labels != -100).sum()
            correct = (shifted_output == shifted_labels).sum()
            acc = correct.float() / to_predict.float()
        else:
            acc = ((output > 0) == labels.bool()).sum().float()
        return acc

    def _edit_status(self, step, loss, l_edit, l_loc, acc, res_p):
        return (
            f"step: {step}".ljust(14)
            + f"loss: {loss.item():.5f}".ljust(18)
            + f"l_edit: {l_edit.item():.5f}".ljust(18)
            + f"l_loc: {l_loc.item():.5f}".ljust(18)
            + f"acc: {acc.item():.2f}".ljust(14)
            + f"norm: {res_p.view(-1).norm().item():.5f}"
        )

    def edit(self, batch, condition=None, detach_history=False):
        edit_model = self.model.eval()
        p0 = list(edit_model.named_parameters())

        if not isinstance(edit_model, higher.patch._MonkeyPatchBase):
            edit_model = make_functional(
                self.model, track_higher_grads=False, in_place=True
            )

        packed_residuals = {}
        opt_params = []
        for n, p in _inner_params(
            edit_model.named_parameters(), self.config.model.inner_params
        ):
            if self.config.ft.rank is not None:
                u = nn.Parameter(
                    torch.randn(p.shape[0], self.config.ft.rank, device=p.device)
                    * self.config.ft.init_std
                )
                v = nn.Parameter(
                    torch.zeros(self.config.ft.rank, p.shape[1], device=p.device)
                )
                res = [u, v]
            else:
                res = [nn.Parameter(torch.zeros_like(p, device=p.device))]

            packed_residuals[n] = res
            opt_params.extend(res)

        assert len(opt_params) == len(self.config.model.inner_params)
        OptClass = getattr(torch.optim, self.config.ft.opt)
        opt = OptClass(opt_params, lr=self.config.edit_lr)

        start_time = time.time()
        for edit_step in range(self.config.ft.max_edit_steps):
            if self.config.ft.time_limit is not None and (
                time.time() - start_time > self.config.ft.time_limit
            ):
                break
            residuals = {
                k: v[0] @ v[1] if len(v) == 2 else v[0]
                for k, v in packed_residuals.items()
            }
            edited_params = [
                p if n not in residuals else p.detach() + residuals[n] for n, p in p0
            ]
            loss, l_edit, l_loc, acc = self._edit_loss(
                edit_model, [p for n, p in p0], edited_params, batch
            )

            if self.config.ft.verbose:
                residual = list(residuals.values())[-1]
                print(
                    self._edit_status(edit_step, loss, l_edit, l_loc, acc, residual),
                    end="\r",
                )

            if acc == 1.0:
                break

            for p, g in zip(opt_params, torch.autograd.grad(loss, opt_params)):
                p.grad = g
            torch.nn.utils.clip_grad_norm_(opt_params, self.config.grad_clip)
            opt.step()
            opt.zero_grad()

        if detach_history:
            new_model = self.model_constructor()
            new_model.load_state_dict(edit_model.state_dict())
            edit_model = new_model
        edit_model.train(self.training)

        return (
            FT(edit_model, self.config, self.model_constructor, self.edit_loss_fn),
            {},
        )
