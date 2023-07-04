import higher
import torch
import torch.nn as nn
from .editable_model import EditableModel
from ..utils import _logits


def fomaml_callback(all_grads):
    return [g.detach() if g is not None else None for g in all_grads]


class ENN(EditableModel):
    def __init__(
        self, model, config, model_constructor, edit_lrs=None, edit_loss_fn=None
    ):
        super().__init__(model, config, model_constructor)

        if edit_lrs is None:
            edit_lrs = nn.Parameter(
                torch.tensor([config.edit_lr] * len(self.config.model.inner_params))
            )
        self.edit_lrs = edit_lrs

        if edit_loss_fn is not None:
            self.edit_loss_fn = edit_loss_fn

        self.grad_callback = fomaml_callback if config.enn.first_order else lambda x: x

    def outer_parameters(self):
        if self.config.no_grad_layers is None:
            return super().outer_parameters()
        else:
            params = [self.edit_lrs]
            for m in self.model.modules():
                if isinstance(m, nn.ModuleList):
                    params.extend(list(m[self.config.no_grad_layers :].parameters()))
            return params

    def get_state_dict(self):
        return self.state_dict()

    def edit(self, batch, condition=None, detach_history=False):
        opt = torch.optim.SGD(
            [
                {"params": p, "lr": None}
                for (n, p) in self.model.named_parameters()
                if n in self.config.model.inner_params
            ]
        )
        with torch.enable_grad(), higher.innerloop_ctx(
            self.model,
            opt,
            override={"lr": list(self.edit_lrs)},
            copy_initial_weights=False,
            track_higher_grads=self.training,
            in_place=True,
        ) as (fmodel, diffopt):
            fmodel.eval()
            for edit_step in range(self.config.enn.n_edit_steps):
                output = _logits(fmodel(**batch))
                loss = self.edit_loss_fn(output, batch["labels"])["nll"]
                diffopt.step(loss, grad_callback=self.grad_callback)

        if not detach_history:
            model_edited = fmodel
        else:
            model_edited = self.model_constructor()
            model_edited.load_state_dict(fmodel.state_dict())
        model_edited.train(self.training)

        return (
            ENN(
                model_edited,
                self.config,
                self.model_constructor,
                edit_lrs=self.edit_lrs,
                edit_loss_fn=self.edit_loss_fn,
            ),
            {},
        )


def test():
    import copy
    import types

    import transformers

    model = transformers.GPT2LMHeadModel.from_pretrained("gpt2")

    config = types.SimpleNamespace()
    config.edit_lr = 0.1
    config.model.inner_params = [
        "transformer.h.9.mlp.c_fc.weight",
        "transformer.h.9.mlp.c_proj.weight",
        "transformer.h.10.mlp.c_fc.weight",
        "transformer.h.10.mlp.c_proj.weight",
        "transformer.h.11.mlp.c_fc.weight",
        "transformer.h.11.mlp.c_proj.weight",
    ]
    config.enn = {"n_edit_steps": 2, "first_order": False}

    enn = ENN(model, config, lambda: copy.deepcopy(model)).cuda()

    x = torch.arange(100).view(5, 20).cuda() + 1000

    edited = enn.edit(x, masks=torch.ones_like(x), labels=x)

    orig_param = [
        p
        for (n, p) in enn.model.named_parameters()
        if n == config.model.inner_params[-1]
    ][0]
    edited_param = [
        p
        for (n, p) in edited.model.named_parameters()
        if n == config.model.inner_params[-1]
    ][0]

    print((orig_param - edited_param).abs().max())
    edited.eval()
    print(
        enn(x, labels=x).loss,
        edited(x, labels=x).loss,
        edited.edit_loss_fn(edited(x).logits, x)["nll"],
    )
    edited.edit_loss_fn(edited(x).logits, x).backward()
    import pdb

    pdb.set_trace()


if __name__ == "__main__":
    with torch.autograd.set_detect_anomaly(True):
        test()
