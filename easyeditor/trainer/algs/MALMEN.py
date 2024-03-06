import time
from typing import Dict, List
# from omegaconf import DictConfig
from torch.nn.utils import clip_grad_norm_
from collections import Counter
import numpy as np
import logging
from .editable_model import EditableModel

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from .malmen.nets import MALMENNet
import math
from tqdm import tqdm
# import wandb

from .malmen.util import (
    get_module,
    get_shape,
    TracerDict,
    cross_entropy,
    kl_div,
    succ_ratios
)

from ..utils import (
    EarlyStopper,
    RunningStatAverager,
    _logits,
    formatted_timestamp,
    safe_backward,
    time_delta_seconds,
)

LOG = logging.getLogger(__name__)


class MALMEN(EditableModel):

    def __init__(
        self, model: nn.Module, config, model_constructor
    ):
        super().__init__(model, config, model_constructor)

        self.shift = False
        if 'gpt' in config.model_name.lower():
            self.shift = True
        elif 'llama' in config.model_name.lower():
            self.shift = True
        elif 'internlm' in config.model_name.lower():
            self.shift = True
        elif 'chatglm' in config.model_name.lower():
            self.shift = True
        elif 'qwen' in config.model_name.lower():
            self.shift = True
        elif 'mistral' in config.model_name.lower():
            self.shift = True

        if not str(self.config.device).startswith('cuda'):
            self.config.device = f'cuda:{self.config.device}'
        
        if config.half:
            self.model.bfloat16()

        for param in self.model.parameters():
            param.requires_grad = False
        
        for i in range(len(config.inner_params)):
            if config.inner_params[i].endswith(".weight"):
                    config.inner_params[i] = config.inner_params[i].replace(".weight", "")
        self.config.inner_params = config.inner_params

        for module_name in config.inner_params:
            module = get_module(self.model, module_name)
            module.weight.requires_grad = True

        shape_counter = Counter()
        self.name2idx = {}
        for module_name in config.inner_params:
            shape = get_shape(get_module(model, module_name))
            self.name2idx[module_name] = shape_counter[shape]
            shape_counter[shape] += 1

        self.net = nn.ModuleDict({
            str(k): MALMENNet(
                *k,
                config.rank,
                config.n_blocks,
                v,
                config.lr
            )
            for k, v in shape_counter.items()
        }).to(config.device)
        
        self.opt = torch.optim.Adam(
            self.net.parameters(),
            config.meta_lr
        )

    def edit_model(
        self,
        param_shifts: Dict[str, torch.FloatTensor],
        is_reverse: bool
    ):
        
        for module_name, param_shift in param_shifts.items():
            module = get_module(self.model, module_name)
            if isinstance(module, nn.Linear):
                param_shift = param_shift.T
            if is_reverse:
                param_shift = - param_shift
            module.weight.data += param_shift.to(module.weight.data.dtype)

    def train(self, batch):
        start = time.time()

        batch_dv = {}  
        
        for item_dict in batch:   
            for key, value in item_dict.items():  
                if key not in batch_dv:  
                    batch_dv[key] = []   
                batch_dv[key].append(value)

        module_kv_map = self.cache(batch_dv["edit_inner"])
        param_shifts = self.predict_param_shifts(module_kv_map)
        self.model.zero_grad()
        
        # gen_loss
        self.edit_model(param_shifts, False)
        edit_time = time.time() - start
        
        gen_losses = []
        for t in batch_dv["edit_rephrase"]:
            logits = self.model(input_ids=t['input_ids'], attention_mask=t['attention_mask'])["logits"]
            loss = cross_entropy(logits, t["labels"], self.shift)
            loss.backward()
            gen_losses += [loss.item()]
        self.edit_model(param_shifts, True)

        # loc_loss
        loc_losses = []
        for t in batch_dv["loc"]:
            with torch.no_grad():
                refer_logits = self.model(input_ids=t['input_ids'], attention_mask=t['attention_mask'])["logits"]

            self.edit_model(param_shifts, False)
            logits = self.model(input_ids=t['input_ids'], attention_mask=t['attention_mask'])["logits"]

            loss = kl_div(
                refer_logits,
                logits,
                t["labels"],
                self.shift
            )

            (self.config.loc_coef * loss).backward()
            self.edit_model(param_shifts, True)
            loc_losses += [loss.item()]
            
        self.update_hypernet(param_shifts, module_kv_map)

        info_dict = {}
        info_dict["gen_loss"] = np.mean(gen_losses)
        info_dict["loc_loss"] = np.mean(loc_losses)
        info_dict["time/edit"] = edit_time

        # LOG.info({
        #     "gen_loss": gen_losses,
        #     "loc_loss": loc_losses
        # })
        return info_dict

    def cache(self, batch) -> Dict[int, Dict[int, Dict[str, torch.Tensor]]]:
        module_kv_map = {}
        for idx, t in enumerate(batch):
            with TracerDict(
                self.model,
                self.config,
                t
            ) as tr:
                logits = self.model(input_ids=t['input_ids'], attention_mask=t['attention_mask'])["logits"]
                cross_entropy(logits, t["labels"], self.shift).backward()
            for module_idx, module_name in enumerate(self.config.inner_params):
                shape = get_shape(get_module(self.model, module_name))
                keys = tr[module_name].keys.to(torch.float32).to(self.config.device)
                values_grad = tr[module_name].values_grad.to(torch.float32).to(self.config.device)
                self.net[str(shape)].normalizer.update(torch.cat((keys, values_grad), -1))
                module_kv_map.setdefault(module_idx, {}).update({idx: {'keys': keys, 'values_grad': values_grad}})
        return module_kv_map

    def predict_param_shifts(self, module_kv_map) -> Dict[str, torch.FloatTensor]:
        
        param_shifts = {}
        for module_idx, module_name in enumerate(self.config.inner_params):

            shape = get_shape(get_module(self.model, module_name))
            net = self.net[str(shape)]
            layer_idx = torch.LongTensor([self.name2idx[module_name]]).to(self.config.device)
            keys = torch.cat([
                module_kv_map[module_idx][idx]["keys"]
                for idx in range(len(module_kv_map[module_idx]))
            ])
            values_grad = torch.cat([
                module_kv_map[module_idx][idx]["values_grad"]
                for idx in range(len(module_kv_map[module_idx]))
            ])
            value_diffs = torch.empty((0, net.value_size), device = self.config.device)
            for start_idx in range(0, keys.shape[0], self.config.editor_batch_size):
                end_idx = start_idx + self.config.editor_batch_size
                with torch.no_grad():
                    pesudo_keys, pesudo_values_grad = net(
                        keys[start_idx:end_idx],
                        values_grad[start_idx:end_idx],
                        layer_idx
                    )
                    coeffs = - net.lr(layer_idx) * (keys[start_idx:end_idx] * pesudo_keys).sum(-1).unsqueeze(-1)
                value_diffs = torch.cat((value_diffs, coeffs * pesudo_values_grad))
            with torch.no_grad():
                mat = keys.T @ keys + net.lamda(layer_idx).exp() * torch.eye(net.key_size, device = self.config.device)
            param_shift = torch.linalg.solve(mat, keys.T @ value_diffs)
            param_shifts[module_name] = param_shift.to(next(self.model.parameters()).device)

        return param_shifts
        
    def update_hypernet(self, param_shifts: Dict[str, torch.FloatTensor], module_kv_map):
        
        self.opt.zero_grad()
        for module_idx, module_name in enumerate(self.config.inner_params):
            shape = get_shape(get_module(self.model, module_name))
            net = self.net[str(shape)]
            layer_idx = torch.LongTensor([self.name2idx[module_name]]).to(self.config.device)
            keys = torch.cat([
                module_kv_map[module_idx][idx]["keys"]
                for idx in range(len(module_kv_map[module_idx]))
            ])
            values_grad = torch.cat([
                module_kv_map[module_idx][idx]["values_grad"]
                for idx in range(len(module_kv_map[module_idx]))
            ])
            module = get_module(self.model, module_name)
            module_grad = module.weight.grad.to(torch.float32).to(self.config.device)
            param_shift = param_shifts[module_name].to(self.config.device)
            if isinstance(module, nn.Linear):
                module_grad = module_grad.T
            with torch.no_grad():
                mat = torch.linalg.solve(keys.T @ keys + net.lamda(layer_idx).exp() * torch.eye(net.key_size, device = self.config.device), module_grad)
                lamda_grad = - net.lamda(layer_idx).exp() * (mat * param_shift).sum()
            value_diffs_grad = keys @ mat
            (lamda_grad * net.lamda(layer_idx)).backward()
            for start_idx in range(0, keys.shape[0], self.config.editor_batch_size):
                end_idx = start_idx + self.config.editor_batch_size
                pesudo_keys, pesudo_values_grad = net(
                    keys[start_idx:end_idx],
                    values_grad[start_idx:end_idx],
                    layer_idx
                )
                coeffs = - net.lr(layer_idx) * (keys[start_idx:end_idx] * pesudo_keys).sum(-1).unsqueeze(-1)
                value_diff = coeffs * pesudo_values_grad
                (value_diffs_grad[start_idx:end_idx] * value_diff).sum().backward()
            
        clip_grad_norm_(
            self.net.parameters(),
            self.config.max_grad_norm
        )
        self.opt.step()  

    def _inline_malmen_valid_log(self, step, stats, start_time, steps):

        elapsed = (time.time() - start_time) / (step + 1)
        prog = f"{step+1}/{steps}".ljust(20)
        edit_acc = f"{stats['ES_val']:<12.5f}"
        gen_acc = f"{stats['GS_val']:<12.5f}"
        loc_acc = f"{stats['LS_val']:<12.5f}"
        
        LOG.info(
            f"Step {prog} edit_acc: {edit_acc} gen_acc: {gen_acc} loc_acc: {loc_acc}"
        )

    def state_dict(self, destination=None, prefix="", keep_vars=False):
        state_dict = self.net.state_dict(prefix=prefix, keep_vars=keep_vars)  # Get default state dict
        return state_dict

    def load_state_dict(self, state_dict, strict: bool = True):
        res = self.net.load_state_dict(state_dict, False)
        return res

    def to(self, device):
        super().to(device)
        self.net.to(device)
        self.model.to(device)

    def valid(self, config, loader, val_set, steps, log: bool = False):
        if steps is None or steps > len(loader):
            steps = len(loader)

        if steps < math.ceil(self.config.n_edits / self.config.batch_size):
            steps = math.ceil(self.config.n_edits / self.config.batch_size)

        if log:
            LOG.info(f"Beginning evaluation for {steps} steps...")
        averager = RunningStatAverager("val")

        start_time = time.time()
        n_edits_batch = []
        for val_step, batch in enumerate(loader):
            if val_step >= steps:
                break
            n_edits_batch.append(batch)
            if (val_step + 1) % math.ceil(self.config.n_edits / self.config.batch_size) == 0 or val_step == steps-1:
            # edit
                batch_dv = {}  
                for item_dict in n_edits_batch:   
                    for key, value in item_dict.items():  
                        if key not in batch_dv:  
                            batch_dv[key] = []   
                        batch_dv[key].append(value)
                n_edits_batch = []

                module_kv_map = self.cache(batch_dv["edit_inner"])
                param_shifts = self.predict_param_shifts(module_kv_map)
                self.edit_model(param_shifts, False)
                edit_succs, gen_succs, loc_succs = [], [], []
                for k, s in zip(
                    ["edit_inner", "edit_rephrase", "loc"],
                    [edit_succs, gen_succs, loc_succs]
                ):
                    for t in batch_dv[k]:
                        with torch.no_grad():
                            logits = self.model(input_ids=t['input_ids'], attention_mask=t['attention_mask'])["logits"]
                        s += succ_ratios(logits, t["labels"], self.shift)
                        
                self.edit_model(param_shifts, True)
                
                info_dict = {}
                info_dict["ES"] = np.mean(edit_succs)
                info_dict["GS"] = np.mean(gen_succs)
                info_dict["LS"] = np.mean(loc_succs)

                averager.add(info_dict)

            if (
                log
                and (val_step + 1) % config.log_interval == 0
            ):
                self._inline_malmen_valid_log(
                    val_step, averager.average(), start_time, steps
                )

        if log:
            self._inline_malmen_valid_log(val_step, averager.average(), start_time, steps)
        elapsed = time.time() - start_time
        stats = averager.average()
        stats["eval_time/elapsed"] = elapsed
        stats["eval_time/average"] = elapsed / steps
        return stats

    def convert_last_zero_to_one_in_mask(mask):
        last_zero_indices = []  
        for i in range(mask.size(0)):  
            row = mask[i]  
            last_zero_idx = (row == 0).nonzero()[-1, 0].item() if (row == 0).any() else -1  
            last_zero_indices.append(last_zero_idx)  
        last_zero_indices = torch.tensor(last_zero_indices, device=mask.device)  
        mask[range(mask.size(0)), last_zero_indices] = 1  
