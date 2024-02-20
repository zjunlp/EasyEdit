from .BaseTrainer import *
import json
import logging
import os
import shutil
import tempfile
import time

import torch
from .losses import kl_loc_loss
from omegaconf import OmegaConf
from torch.utils.data import Dataset
from .utils import (
    EarlyStopper,
    RunningStatAverager,
    _logits,
    formatted_timestamp,
    safe_backward,
    time_delta_seconds,
)

LOG = logging.getLogger(__name__)


class MultiTaskTrainer(BaseTrainer):
    def __init__(self, config, train_set: Dataset, val_set: Dataset):
        super().__init__(config, train_set, val_set)
        if isinstance(train_set.tok, GPT2Tokenizer) or isinstance(train_set.tok, GPT2TokenizerFast):
            self.model.model.resize_token_embeddings(len(train_set.tok))
            self.model.model.lm_head.weight.data[-1, :] = self.model.model.lm_head.weight.data.mean(0)
        if hasattr(self.model, "edit_lrs") and not self.config.eval_only:
            self.lr_opt = self.OptimizerClass([self.model.edit_lrs], config.lr_lr)
            if self.archive is not None:
                self.lr_opt.load_state_dict(self.archive["lr_opt"])
        else:
            self.lr_opt = None

    def edit_step(self, batch, training: bool):
        self.model.train(training)
        self.original_model.train(training)

        with torch.no_grad():
            if batch["metric_kwargs"] is None:
                base_logits = self.model(**batch["loc"])
            else:
                base_logits = self.model(
                    input_ids=batch["loc"]["input_ids"].to(self.config.device),
                    attention_mask=batch["loc"]["attention_mask"].to(self.config.device),
                    )

        # Do the edit
        start = time.time()
        if batch["metric_kwargs"] is None:
            if "cond" in batch:
                edited_model, model_info = self.model.edit(batch["edit_inner"], batch["cond"])
            else:
                edited_model, model_info = self.model.edit(batch["edit_inner"])
        else:
            kwargs = dict(
                inner_sent=batch["metric_kwargs"]["inner_target"],
                same_mask=batch["metric_kwargs"]["same_mask"],
                unlikelihood=True,
            )
            if "cond" in batch:
                edited_model, model_info = self.model.edit(batch["edit_inner"], batch["cond"], **kwargs)
            else:
                edited_model, model_info = self.model.edit(batch["edit_inner"], **kwargs)
            
        edit_time = time.time() - start

        with torch.set_grad_enabled(training):
            # Editing loss
            post_edit_logits = edited_model(**batch["edit_inner"])
            if batch["metric_kwargs"] is None:
                l_edit = self.model.edit_loss_fn(
                    self.config, post_edit_logits, batch["edit_inner"]["labels"],
                )["nll"]
            else:
                l_edit = self.model.edit_loss_fn(
                    self.config, post_edit_logits, batch["edit_inner"]["labels"], **kwargs
                )["nll"]

            # Locality loss
            if batch["metric_kwargs"] is None:
                post_base_logits = edited_model(**batch['loc'])
            else:
                post_base_logits = edited_model(
                    input_ids=batch["loc"]["input_ids"].to(self.config.device),
                    attention_mask=batch["loc"]["attention_mask"].to(self.config.device),
                    labels=batch["loc"]["labels"].to(self.config.device),
                    )
            kl_mask = batch["loc"].get(
                "decoder_attention_mask", batch["loc"]["attention_mask"]
            )
            if kl_mask.size(1) != base_logits.size(1):
                base_logits = base_logits[:, -kl_mask.size(1):]
                post_base_logits = post_base_logits[:, -kl_mask.size(1):]
            l_loc = kl_loc_loss(base_logits.detach(), post_base_logits, mask=kl_mask)

        l_total_edit = self.config.cedit * l_edit + self.config.cloc * l_loc

        if training:
            safe_backward(
                l_total_edit, self.model.outer_parameters(), self.config.accumulate_bs, allow_unused=True if
                self.config.alg=='MEND' and self.config.model_parallel else False
            )

        # Collect some useful metrics
        with torch.no_grad():
            if batch["metric_kwargs"] is None:
                post_edit_dict = self.model.edit_loss_fn(
                    self.config, post_edit_logits, batch["edit_inner"]["labels"]
                )
            else:
                post_edit_dict = self.model.edit_loss_fn(
                    self.config, post_edit_logits, batch["edit_inner"]["labels"], **kwargs
                )
            post_loc_dict = self.model.loc_loss_fn(
                self.config, post_base_logits, batch["loc"]["labels"]
            )
            pre_loc_dict = self.model.loc_loss_fn(
                self.config, base_logits, batch["loc"]["labels"]
            )

        # text loc
        post_base_logits_softmax_top_k = torch.topk(torch.nn.functional.softmax(post_base_logits, dim=-1), k=1, dim=-1).indices
        base_logits_softmax_top_k = torch.topk(torch.nn.functional.softmax(base_logits, dim=-1), k=1, dim=-1).indices

        info_dict = {}
        info_dict["loss/edit"] = l_edit.item()
        info_dict["loss/loc"] = l_loc.item()
        info_dict["edit/acc"] = post_edit_dict["acc"].item()
        info_dict["loc/acc"] = sum(post_base_logits_softmax_top_k.view(-1) == base_logits_softmax_top_k.view(-1))/post_base_logits_softmax_top_k.view(-1).shape[0]
        info_dict["edit/log_prob"] = post_edit_dict["log_prob"].item()
        info_dict["edit/prob"] = post_edit_dict["prob"].item()
        info_dict["time/edit"] = edit_time

        # Base loss
        if self.config.train_base:
            with torch.no_grad():
                original_logits = _logits(self.original_model(**batch["loc"]))
                original_loc_dict = self.model.loc_loss_fn(
                    original_logits, batch["loc"]["labels"]
                )

            base_logits = self.model(**batch["loc"])
            l_base = kl_loc_loss(
                original_logits.detach(), base_logits, mask=kl_mask.detach()
            )

            if training:
                safe_backward(
                    l_base,
                    self.model.outer_parameters(),
                    self.config.accumulate_bs,
                    allow_unused=True,
                )

            info_dict["loss/base"] = l_base.item()
            info_dict["nll/original"] = original_loc_dict["nll"].item()
            info_dict["acc/original"] = original_loc_dict["acc"].item()
            info_dict["n_tokens/original"] = original_loc_dict["n_tokens"]
        else:
            l_base = torch.tensor(0.0)

        l_total = l_total_edit + self.config.cbase * l_base

        info_dict["loss/total"] = l_total.item()
        info_dict["loss/total_edit"] = l_total_edit.item()
        info_dict["memory/alloc_max"] = torch.cuda.max_memory_allocated()
        info_dict["memory/res_max"] = torch.cuda.max_memory_reserved()
        info_dict = {**info_dict, **model_info}

        return l_total, l_edit, l_loc, l_base, info_dict

    def train_step(self, batch):
        l_total, l_edit, l_loc, l_base, info_dict = self.edit_step(
            batch, training=True
        )

        if self.global_iter > 0 and self.global_iter % self.config.accumulate_bs == 0:
            grad = torch.nn.utils.clip_grad_norm_(
                self.model.outer_parameters(),
                self.config.grad_clip,
                error_if_nonfinite=True,
            )
            info_dict["grad"] = grad.item()

            self.opt.step()
            self.opt.zero_grad()

            if self.lr_opt is not None:
                self.lr_opt.step()
                self.lr_opt.zero_grad()

                for lr_idx, lr in enumerate(self.model.edit_lrs):
                    info_dict[f"lr/lr{lr_idx}"] = lr.item()

        return info_dict

    def _inline_validation_log(self, step, stats, start_time, steps):
        elapsed = (time.time() - start_time) / (step + 1)
        prog = f"{step+1}/{steps}".ljust(20)
        acc = f"{stats['edit/acc_val']:<12.5f}"
        loc_acc = f"{stats['loc/acc_val']:<12.5f}"

        LOG.info(
            f"Step {prog} edit_acc: {acc} loc_acc: {loc_acc} it_time: {elapsed:.4f}"
        )

    def validate(self, steps=None, log: bool = False):
        if steps is None or steps > len(self.val_set):
            steps = len(self.val_set)

        if log:
            LOG.info(f"Beginning evaluation for {steps} steps...")
        averager = RunningStatAverager("val")

        start_time = time.time()
        for val_step, batch in enumerate(self.val_loader):
            if val_step >= steps:
                break
            _, _, _, _, info_dict = self.edit_step(batch, training=False)
            averager.add(info_dict)

            if (
                log
                and (val_step + 1) % self.config.log_interval == 0
            ):
                self._inline_validation_log(
                    val_step, averager.average(), start_time, steps
                )

        if log:
            self._inline_validation_log(val_step, averager.average(), start_time, steps)
        elapsed = time.time() - start_time
        stats = averager.average()
        stats["eval_time/elapsed"] = elapsed
        stats["eval_time/average"] = elapsed / steps

        return stats
