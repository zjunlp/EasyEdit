from .BaseTrainer import *
import json
import logging
import os
import shutil
import tempfile
import time
from tqdm import tqdm

import torch
from .losses import kl_loc_loss, es
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


class PEREditTrainer(BaseTrainer):
    def __init__(self, config, train_set: Dataset, val_set: Dataset):
        super().__init__(config, train_set, val_set)

        if hasattr(self.model, "edit_lrs") and not self.config.eval_only:
            self.lr_opt = self.OptimizerClass([self.model.edit_lrs], config.lr_lr)
            if self.archive is not None:
                self.lr_opt.load_state_dict(self.archive["lr_opt"])
        else:
            self.lr_opt = None
            
        if 'gpt' in self.config.model_class.lower():
            self.model.model.resize_token_embeddings(len(val_set.tok))

    def edit_step(self, batch, training: bool):
        self.model.train(training)
        self.original_model.train(training)

        with torch.no_grad():
            base_logits = self.model(**batch["loc"])
        
        # Do the edit
        start = time.time()
        edited_model, model_info = self.model.edit(batch["cond"], personality=True)# batch["edit_inner"]
        edit_time = time.time() - start

        with torch.set_grad_enabled(training):
            # Editing loss
            pre_edit_logits = self.model(**batch["edit_outer"])
            post_edit_logits = edited_model(**batch["edit_outer"])
            
            kwargs = dict(
                pre_edit_logits=pre_edit_logits,
                post_edit_logits=post_edit_logits.detach(),
                inner_sent=batch["inner_per"],
                outer_sent=batch["outer_per"],
                same_mask=batch["same_mask"],
                unlikelihood=True,
                q_mask=batch["edit_outer"]["q_mask"] 
            )
            
            l_edit = self.model.edit_loss_fn(
                self.config,
                post_edit_logits,
                batch["edit_outer"]["labels"],
                **kwargs,
            )["nll"]
            
            # Locality loss
            post_base_logits = edited_model(**batch['loc'])
            l_loc = kl_loc_loss(base_logits.detach(), post_base_logits, mask=batch["kl_mask"])

        l_total_edit = self.config.cedit * l_edit + self.config.cloc * l_loc

        if training:
            safe_backward(
                l_total_edit, self.model.outer_parameters(), self.config.accumulate_bs, allow_unused=True if
                self.config.alg=='MEND' else False
            )

        # Collect some useful metrics
        with torch.no_grad():
            # post_edit_dict = self.model.edit_loss_fn(
            #     self.config, post_edit_logits, batch["edit_inner"]["labels"]
            # )
            # post_loc_dict = self.model.loc_loss_fn(
            #     self.config, post_base_logits, batch["loc"]["labels"]
            # )
            # pre_loc_dict = self.model.loc_loss_fn(
            #     self.config, base_logits, batch["loc"]["labels"]
            # )
            
            es_result = es(pre_edit_logits, post_edit_logits, batch["edit_outer"]["labels"], same_per_mask=batch["same_mask"], q_mask=batch["edit_outer"]["q_mask"]),


        info_dict = {}
        info_dict["loss/edit"] = l_edit.item()
        info_dict["loss/loc"] = l_loc.item()
        info_dict["time/edit"] = edit_time
        for k, v in es_result[0].items():
            if isinstance(v, torch.Tensor):
                info_dict[k] = v.cpu().item()
            else:
                info_dict[k] = float(v)

        # Base loss
        if self.config.train_base:
            with torch.no_grad():
                original_logits = _logits(self.original_model(**batch["loc"]))
                original_loc_dict = self.model.loc_loss_fn(
                    original_logits, batch["loc"]["labels"]
                )

            base_logits = self.model(**batch["loc"])
            l_base = kl_loc_loss(
                original_logits.detach(), base_logits, mask=batch["kl_mask"].detach()
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
        es = f"{stats['acc_per_val']:<12.5f}"
        dd = f"{stats['loss/loc_val']:<12.5f}"

        LOG.info(
            f"Step {prog} edit: es: {es}, dd: {dd}, it_time: {elapsed:.4f}"
        )

    def validate(self, steps=None, log: bool = False):
        if steps is None or steps > len(self.val_set):
            steps = len(self.val_set)

        if log:
            LOG.info(f"Beginning evaluation for {steps} steps...")
        averager = RunningStatAverager("val")

        start_time = time.time()
        for val_step, batch in enumerate(tqdm(self.val_loader)):
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
