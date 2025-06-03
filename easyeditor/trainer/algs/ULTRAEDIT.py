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

def pad_tensor(tensor, target_length, dim=0, padding_value=0):

    tensor_length = tensor.size(dim)
    if tensor_length >= target_length:
        return tensor.narrow(dim, 0, target_length)
    else:
        padding = target_length - tensor_length
        pad_shape = list(tensor.shape)
        pad_shape[dim] = padding
        pad_tensor = torch.full(pad_shape, padding_value, dtype=tensor.dtype, device=tensor.device)
        mask = torch.cat([torch.ones(tensor_length, dtype=torch.float32, device=tensor.device),
                          torch.zeros(padding, dtype=torch.float32, device=tensor.device)], dim=0)
        return torch.cat([tensor, pad_tensor], dim=dim)


class ULTRAEDIT(EditableModel):

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

        self.lifelong_normalizer = nn.ModuleDict({
            str(k): RunningMeanStd(
                k[0]+k[1],
            )
            for k, v in self.shape_counter.items()
        }).to(config.editor_device)
        
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
                self.lifelong_normalizer[str(shape)].update(torch.cat((keys, values_grad), -1))
                module_kv_map.setdefault(module_idx, {}).update({idx: {'keys': keys, 'values_grad': values_grad}})
        return module_kv_map

    def predict_param_shifts(self, module_kv_map) -> Dict[str, torch.FloatTensor]:
        
        param_shifts = {}
        for module_idx, module_name in enumerate(self.config.inner_params):

            shape = get_shape(get_module(self.model, module_name))

            lifelong_normalizer = self.lifelong_normalizer[str(shape)]
            hidden_states = torch.cat([
                module_kv_map[module_idx][idx]["keys"]
                for idx in range(len(module_kv_map[module_idx]))
            ])
            values_grad = torch.cat([
                module_kv_map[module_idx][idx]["values_grad"]
                for idx in range(len(module_kv_map[module_idx]))
            ])
            v_feature = torch.empty((0, shape[1]), device = self.config.editor_device)
            for start_idx in range(0, hidden_states.shape[0], self.config.editor_batch_size):
                end_idx = start_idx + self.config.editor_batch_size
                hidden_states_once = pad_tensor(hidden_states[start_idx:end_idx], self.config.editor.batch_size, 0)
                values_grad_once = pad_tensor(values_grad[start_idx:end_idx], self.config.editor.batch_size, 0)
                with torch.no_grad():
                    z_feature = torch.cat((hidden_states_once, values_grad_once), -1)

                    z_feature = lifelong_normalizer(z_feature)
                    (hidden_states_hat, pesudo_values_hat) = z_feature.split([shape[0], shape[1]], -1)
                
                    coeffs = - self.config.editor.lr*(hidden_states_hat * hidden_states_hat).sum(-1).unsqueeze(-1)
                v_feature = torch.cat((v_feature, coeffs * pesudo_values_hat))
            with torch.no_grad():
                mat = hidden_states.T @ hidden_states + torch.eye(shape[0], device=self.config.editor_device)
            v_feature = v_feature[:hidden_states.shape[0], :]
            param_shift = torch.linalg.solve(mat, hidden_states.T @ v_feature)
            param_shifts[module_name] = param_shift.to(next(self.model.parameters()).device)

        return param_shifts
        
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
