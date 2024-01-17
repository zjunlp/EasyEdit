import copy
import logging
from collections import defaultdict

import higher
import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
from collections import deque
from higher.patch import (
    _MonkeyPatchBase,
    _torch,
    _typing,
    _utils,
    buffer_sync,
    make_functional,
)
from .patch import monkeypatch as _make_functional

from . import local_nn
from .editable_model import EditableModel
from .hooks import hook_model
from ..utils import _inner_params, _logits

LOG = logging.getLogger(__name__)


def update_counter(x, m, s, k):
    new_m = m + (x - m) / k
    new_s = s + (x - m) * (x - new_m)

    return new_m, new_s


class GradientTransform(nn.Module):
    def __init__(self, x_dim: int, delta_dim: int, cfg, n_modes=None):
        super().__init__()

        self.x_dim = x_dim
        self.delta_dim = delta_dim
        self.cfg = cfg
        if cfg.combine and (cfg.one_sided or cfg.x_only or cfg.delta_only):
            raise ValueError("cfg.combine cannot be used with one-sided MEND variants")

        self.norm_init = False
        self.register_buffer("u_mean", torch.full((x_dim,), float("nan")))
        self.register_buffer("v_mean", torch.full((delta_dim,), float("nan")))
        self.register_buffer("u_std", torch.full((x_dim,), float("nan")))
        self.register_buffer("v_std", torch.full((delta_dim,), float("nan")))
        self.register_buffer("u_s", torch.full((x_dim,), float("nan")))
        self.register_buffer("v_s", torch.full((delta_dim,), float("nan")))
        self.register_buffer("k", torch.full((1,), float("nan")))

        MlpClass = getattr(local_nn, cfg.mlp_class)
        LOG.info(f"Building Gradient Transform with MLP class {MlpClass}")

        def delta_net():
            return MlpClass(
                delta_dim,
                delta_dim,
                delta_dim * 2,
                cfg.n_hidden,
                init=cfg.init,
                act=cfg.act,
                rank=cfg.rank,
                n_modes=n_modes,
            )

        def x_net():
            return MlpClass(
                x_dim,
                x_dim,
                x_dim * 2,
                cfg.n_hidden,
                init=cfg.init,
                act=cfg.act,
                rank=cfg.rank,
                n_modes=n_modes,
            )

        def combined_net():
            return MlpClass(
                delta_dim + x_dim,
                delta_dim + x_dim,
                (delta_dim + x_dim) * 2,
                cfg.n_hidden,
                init=cfg.init,
                act=cfg.act,
                rank=cfg.rank,
                n_modes=n_modes,
            )

        def ID():
            return lambda x, mode=None: x

        if cfg.combine:
            self.mlp = combined_net()
        elif cfg.one_sided:
            if x_dim > delta_dim:
                self.mlp1, self.mlp2 = ID(), delta_net()
            else:
                self.mlp1, self.mlp2 = x_net(), ID()
        elif cfg.x_only:
            self.mlp1, self.mlp2 = x_net(), ID()
        elif cfg.delta_only:
            self.mlp1, self.mlp2 = ID(), delta_net()
        else:
            self.mlp1, self.mlp2 = x_net(), delta_net()

    def forward(self, u, v, param_idx=None):
        u, v = u.to(torch.float32), v.to(torch.float32)

        u_ = u.view(-1, u.shape[-1])
        v_ = v.view(-1, v.shape[-1])

        nz_mask = (u_ != 0).any(-1) * (v_ != 0).any(
            -1
        )  # Skip batch elements with zero grad
        u_ = u_[nz_mask]
        v_ = v_[nz_mask]

        if self.training:
            for idx in range(u_.shape[0]):
                if not self.norm_init:
                    self.u_mean = u_[idx].clone().detach()
                    self.v_mean = v_[idx].clone().detach()
                    self.u_s.zero_()
                    self.v_s.zero_()
                    self.k[:] = 1
                    self.norm_init = True
                else:
                    self.k += 1
                    self.u_mean, self.u_s = update_counter(
                        u_[idx], self.u_mean, self.u_s, self.k
                    )
                    self.v_mean, self.v_s = update_counter(
                        v_[idx], self.v_mean, self.v_s, self.k
                    )

            if self.k < 2:
                raise RuntimeError(
                    f"Can't perform normalization with only {self.k} samples so far"
                )
            self.u_std = (self.u_s / (self.k - 1)) ** 0.5
            self.v_std = (self.v_s / (self.k - 1)) ** 0.5

        if self.cfg.norm:
            u_input = (u_ - self.u_mean) / (self.u_std + 1e-7)
            v_input = (v_ - self.v_mean) / (self.v_std + 1e-7)
        else:
            u_input = u_
            v_input = v_

        if self.cfg.combine:
            output = self.mlp(torch.cat((u_input, v_input), -1), mode=param_idx)
            out1, out2 = output.split([u.shape[-1], v.shape[-1]], -1)
            return out1, out2
        else:
            return self.mlp1(u_input, mode=param_idx), self.mlp2(
                v_input, mode=param_idx
            )


class MEND(EditableModel):
    def get_shape(self, p):
        # We need to flip the shapes since OpenAI gpt2 uses convs instead of linear
        return (
            p.shape
            if isinstance(self.model, transformers.GPT2LMHeadModel)
            else (p.shape[1], p.shape[0])
        )

    def __init__(self, model, config, model_constructor, mend=None, edit_lrs=None):
        super().__init__(model, config, model_constructor)

        if not str(self.config.device).startswith('cuda'):
            self.config.device = f'cuda:{self.config.device}'

        if edit_lrs is None:
            edit_lrs = nn.Parameter(
                torch.tensor([config.edit_lr] * len(self.config.inner_params))
            )
        self.edit_lrs = edit_lrs

        if not hasattr(self.model, "handles"):
            hook_model(self.model, self.config.inner_params)
            LOG.info(f"Hooked {len(self.model.handles)//2} modules")

        if config.shared:
            shape_dict = defaultdict(list)
            for n, p in _inner_params(
                model.named_parameters(), self.config.inner_params
            ):
                shape_dict[self.get_shape(p)].append(n)
            self.shape_dict = shape_dict

        if mend is None:
            if not config.shared:
                self.mend = nn.ModuleDict(
                    {
                        n.replace(".", "#"): GradientTransform(
                            *self.get_shape(p), config
                        )
                        for (n, p) in _inner_params(
                            model.named_parameters(), self.config.inner_params
                        )
                    }
                )
            else:
                self.mend = nn.ModuleDict(
                    {
                        str(tuple(s)): GradientTransform(
                            *s, config, len(shape_dict[s])
                        )
                        for s in shape_dict.keys()
                    }
                )
            if self.config.model_parallel:
                self.mend.to(deque(self.model.parameters(), maxlen=1)[0].device)
            else:
                self.mend.to(self.config.device)
        else:
            self.mend = mend

    def state_dict(self, destination=None, prefix="", keep_vars=False):
        state_dict = super().state_dict(
            prefix=prefix, keep_vars=keep_vars
        )  # Get default state dict
        model_keys = self.model.state_dict(
            prefix=prefix, keep_vars=keep_vars
        ).keys()  # Remove model params
        for k in model_keys:
            del state_dict[f"model.{k}"]
        state_dict["model_config"] = self.model.config  # Include model config
        return state_dict

    def load_state_dict(self, state_dict, strict: bool = True):
        config = state_dict["model_config"]
        del state_dict["model_config"]
        if config != self.model.config:
            LOG.info("Loaded model config doesn't match current model config.")
            LOG.info(f"Loaded: {config}")
            LOG.info(f"Current: {self.model.config}")

        res = super().load_state_dict(state_dict, False)
        # We should only have missing keys for the model, and no unexpected keys
        assert (
            len([k for k in res.missing_keys if not k.startswith("model.")]) == 0
        ), "Should only have missing keys for model, got " + str(
            [k for k in res.missing_keys if not k.startswith("model.")]
        )
        assert len(res.unexpected_keys) == 0, "Shouldn't have any unexpected keys"
        return res

    def forward(self, *inputs, **kwargs):
        if 'minigpt4' in self.config.model_name.lower() or 'blip' in self.config.model_name.lower():
            outputs = self.model(*inputs, **kwargs)
        elif 'gpt' in self.config.model_name.lower():
            outputs = _logits(self.model(input_ids=kwargs['input_ids'], attention_mask=kwargs['attention_mask']))
            # outputs = outputs[:, -kwargs['labels'].shape[-1]:, :]
        elif 'llama' in self.config.model_name.lower():
            outputs = _logits(self.model(input_ids=kwargs['input_ids'], attention_mask=kwargs['attention_mask']))
            # outputs = outputs[:, -kwargs['labels'].shape[-1]:, :]
        elif 'chatglm2' in self.config.model_name.lower():
            outputs = _logits(self.model(input_ids=kwargs['input_ids'], attention_mask=kwargs['attention_mask']))
            # outputs = outputs[:, -kwargs['labels'].shape[-1]:, :]
        elif 'internlm' in self.config.model_name.lower():
            outputs = _logits(self.model(input_ids=kwargs['input_ids'], attention_mask=kwargs['attention_mask']))
            # outputs = outputs[:, -kwargs['labels'].shape[-1]:, :]
        elif 'qwen' in self.config.model_name.lower():
            outputs = _logits(self.model(input_ids=kwargs['input_ids'], attention_mask=kwargs['attention_mask']))
            # outputs = outputs[:, -kwargs['labels'].shape[-1]:, :]
        elif 'mistral' in self.config.model_name.lower():
            outputs = _logits(self.model(input_ids=kwargs['input_ids'], attention_mask=kwargs['attention_mask']))
            # outputs = outputs[:, -kwargs['labels'].shape[-1]:, :]
        else:
            outputs = _logits(self.model(**kwargs))
        return outputs
    def outer_parameters(self):
        return list(self.mend.parameters()) + [self.edit_lrs]

    def edit(self, batch, condition=None, detach_history=False, return_factors=False):
        if 'minigpt4' in self.config.model_name.lower() or 'blip' in self.config.model_name.lower():
            outputs = self.model(batch)        
            if not isinstance(outputs, torch.Tensor):
                # batch_labels = outputs.labels
                outputs = outputs.logits
            loss = self.edit_loss_fn(self.config, outputs, batch["labels"])["nll"]   # TODO Check whether needs to shift          
        elif 'gpt' in self.config.model_name.lower():
            outputs = _logits(self.model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask']))
            # outputs = outputs[:, -batch['labels'].shape[-1]:, :]
            loss = self.edit_loss_fn(self.config, outputs, batch["labels"])["nll"]
        elif 'llama' in self.config.model_name.lower():
            outputs = _logits(self.model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask']))
            # outputs = outputs[:, -batch['labels'].shape[-1]:, :]
            loss = self.edit_loss_fn(self.config, outputs, batch["labels"])["nll"]
        elif 'baichuan' in self.config.model_name.lower():
            outputs = _logits(self.model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask']))
            # outputs = outputs[:, -batch['labels'].shape[-1]:, :]
            loss = self.edit_loss_fn(self.config, outputs, batch["labels"])["nll"] 
        elif 'chatglm2' in self.config.model_name.lower():
            outputs = _logits(self.model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask']))
            # outputs = outputs[:, -batch['labels'].shape[-1]:, :]
            loss = self.edit_loss_fn(self.config, outputs, batch["labels"])["nll"]            
        elif 'internlm' in self.config.model_name.lower():
            outputs = _logits(self.model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask']))
            # outputs = outputs[:, -batch['labels'].shape[-1]:, :]
            loss = self.edit_loss_fn(self.config, outputs, batch["labels"])["nll"]  
        elif 'qwen' in self.config.model_name.lower():
            outputs = _logits(self.model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask']))
            # outputs = outputs[:, -batch['labels'].shape[-1]:, :]
            loss = self.edit_loss_fn(self.config, outputs, batch["labels"])["nll"]         
        elif 'mistral' in self.config.model_name.lower():
            outputs = _logits(self.model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask']))
            # outputs = outputs[:, -batch['labels'].shape[-1]:, :]
            loss = self.edit_loss_fn(self.config, outputs, batch["labels"])["nll"]  
        else:
            outputs = _logits(self.model(**batch))
            loss = self.edit_loss_fn(self.config, outputs, batch["labels"])["nll"]

        names = set([n for n, p in self.model.named_parameters()])
        pset = set(self.config.inner_params)
        for p in pset:
            assert p in names, f"inner param {p} not in model"

        loss.backward()

        if self.config.shared:
            param_idx = (
                lambda n, p: self.shape_dict[self.get_shape(p)].index(n)
                if self.config.shared
                else None
            )  # noqa: E731
            transformed_factors = {
                n: self.mend[str(tuple(self.get_shape(p)))](
                    p.__x__, p.__delta__, param_idx(n, p)
                )
                for n, p in _inner_params(
                    self.model.named_parameters(), self.config.inner_params
                )
            }
        else:
            transformed_factors = {
                n: self.mend[n.replace(".", "#")](p.__x__, p.__delta__)
                for n, p in _inner_params(
                    self.model.named_parameters(), self.config.inner_params
                )
            }

        # Should be bi,bj->ji for nn.Linear, but GPT2 uses Conv1d instead...
        if isinstance(self.model, transformers.GPT2LMHeadModel):
            targ = "ij"
        else:
            targ = "ji"
        mean_grads = {
            n: torch.einsum(f"bi,bj->{targ}", x, delta)
            for n, (x, delta) in transformed_factors.items()
        }

        info_dict = {}
        if return_factors:
            info_dict["factors"] = transformed_factors
        idx = 0
        for n, p in _inner_params(
            self.model.named_parameters(), self.config.inner_params
        ):
            info_dict[f"grad/true_mag{idx}"] = p.grad.norm(2).item()
            info_dict[f"grad/pseudo_mag{idx}"] = mean_grads[n].norm(2).item()
            info_dict[f"grad/true_std{idx}"] = p.grad.std().item()
            info_dict[f"grad/pseudo_std{idx}"] = mean_grads[n].std().item()
            info_dict[f"grad/diff{idx}"] = (p.grad - mean_grads[n]).norm(2).item()
            info_dict[f"grad/cos{idx}"] = F.cosine_similarity(
                p.grad.reshape(-1), mean_grads[n].reshape(-1), dim=0
            ).item()
            idx += 1

        self.model.zero_grad()

        assert len(self.edit_lrs) == len(list(mean_grads.items()))
        updates = {n: lr * g for lr, (n, g) in zip(self.edit_lrs, mean_grads.items())}

        edited_model = self.model
        if not isinstance(edited_model, higher.patch._MonkeyPatchBase):
            if 'minigpt4' in self.config.model_name.lower() or 'blip' in self.config.model_name.lower():
                edited_model = _make_functional(edited_model, in_place=True)
            else:
                edited_model = monkeypatch(edited_model, in_place=True)

        new_params = []
        for n, p in edited_model.named_parameters():
            if n in pset:
                new_params.append(p + updates[n].to(p.dtype))
            else:
                new_params.append(p)

        edited_model.update_params(new_params)

        if detach_history:
            new_model = self.model_constructor()
            new_model.load_state_dict(edited_model.state_dict())
            edited_model = new_model

        return (
            MEND(
                edited_model,
                self.config,
                self.model_constructor,
                self.mend,
                edit_lrs=self.edit_lrs,
            ),
            info_dict,
        )


if __name__ == "__main__":
    import types

    model = transformers.GPT2LMHeadModel.from_pretrained("gpt2")

    config = types.SimpleNamespace()
    config.inner_params = [
        "transformer.h.9.mlp.c_fc.weight",
        "transformer.h.9.mlp.c_proj.weight",
        "transformer.h.10.mlp.c_fc.weight",
        "transformer.h.10.mlp.c_proj.weight",
        "transformer.h.11.mlp.c_fc.weight",
        "transformer.h.11.mlp.c_proj.weight",
    ]
    config.edit_lr = 0.0001

    # config.mend = types.SimpleNamespace()
    config.n_hidden = 1
    config = config.__dict__

    mend = MEND(model, config, lambda: copy.deepcopy(model)).cuda()
    import pdb

    pdb.set_trace()
    mend.load_state_dict(torch.load("test_state.pt"))
    x = torch.arange(20).view(1, 20).cuda() + 1000
    orig_logits = mend(x)
    edited = mend.edit(x, masks=torch.ones_like(x), labels=x)
    post_logits = mend(x)

    assert torch.allclose(orig_logits, post_logits)

    orig_param = [
        p
        for (n, p) in mend.model.named_parameters()
        if n == config.inner_params[-1]
    ][0]
    edited_param = [
        p
        for (n, p) in edited.model.named_parameters()
        if n == config.inner_params[-1]
    ][0]

    LOG.info((orig_param - edited_param).abs().max())
    edited.eval()
    LOG.info(
        mend(x, labels=x).loss,
        edited(x, labels=x).loss,
        edited.edit_loss_fn(edited(x).logits, x)["nll"],
    )
    edited2 = edited.edit(x, masks=torch.ones_like(x), labels=x)
    LOG.info(
        mend(x, labels=x).loss, edited(x, labels=x).loss, edited2(x, labels=x).loss
    )


def monkeypatch(
    module: _torch.nn.Module,
    device: _typing.Optional[_torch.device] = None,
    copy_initial_weights: bool = True,
    track_higher_grads: bool = True,
    in_place: bool = False,
) -> _MonkeyPatchBase:
    r"""Create a monkey-patched stateless version of a module.
    This function produces a monkey-patched version of a module, and returns a
    copy of its parameters for use as fast weights. Where the original module
    or any of its submodules have state (e.g. batch norm), this will be copied
    too, but further updates (e.g. during inner loop training) will cause these
    to diverge without changing the state of the original module.
    Args:
        module: a ``torch.nn.Module`` subclass instance.
        device (optional): a device to cast the fast weights and state to.
        copy_initial_weights: if True, the weights of the patched module are
            copied to form the initial weights of the patched module, and thus
            are not part of the gradient tape when unrolling the patched module.
            If this is set to False, the actual module weights will be the
            initial weights of the patched module. This is useful when doing
            MAML, for example.
        track_higher_grads: if True, during unrolled optimization the graph be
            retained, and the fast weights will bear grad funcs, so as to permit
            backpropagation through the optimization process. Setting this to
            False allows ``monkeypatch`` to be used in "test mode", without
            potentially tracking higher order gradients. This can be useful when
            running the training loop at test time, e.g. in k-shot learning
            experiments, without incurring a significant memory overhead.
    Returns:
        ``fmodule``: a "stateless" version of the original module, for which calls
        to forward take the additional kwarg-only parameter ``params``, which
        should be a list of torch tensors requiring gradients, ideally
        provided by this function (see below) or by an update step from one
        of the optimizers in ``higher.optim``.
    """

    def encapsulator(fmodule: _MonkeyPatchBase, module: _torch.nn.Module) -> None:
        if copy_initial_weights and not in_place:
            params = _utils.get_func_params(module, device=device)
        elif in_place:
            params = [
                p if device is None else p.to(device) for p in module.parameters()
            ]
        else:  # Standard behavior
            params = [
                p.clone() if device is None else p.clone().to(device)
                for p in module.parameters()
            ]
        buffer_sync(module, fmodule, device)
        fmodule.update_params(params)

    fmodule = make_functional(module, encapsulator=encapsulator)
    fmodule.track_higher_grads = track_higher_grads

    return fmodule
