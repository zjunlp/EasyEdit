import time
from typing import Dict, List
from torch.nn.utils import clip_grad_norm_
from collections import Counter
import numpy as np
import logging
from ...trainer.algs.editable_model import EditableModel

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import math
from tqdm import tqdm
# import wandb

from ...trainer.algs.malmen.util import (
    get_module,
    get_shape,
    TracerDict,
    cross_entropy,
    kl_div,
    succ_ratios
)

from ...trainer.algs.malmen.nets import RunningMeanStd
from ...trainer.utils import (
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
            self.model.half()

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
            for k, v in shape_counter.items()
        }).to(self.config.device)
        
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
            v_feature = torch.empty((0, shape[1]), device = self.config.device)
            for start_idx in range(0, hidden_states.shape[0], self.config.editor_batch_size):
                end_idx = start_idx + self.config.editor_batch_size
                hidden_states_once = pad_tensor(hidden_states[start_idx:end_idx], self.config.editor_batch_size, 0)
                values_grad_once = pad_tensor(values_grad[start_idx:end_idx], self.config.editor_batch_size, 0)
                with torch.no_grad():
                    z_feature = torch.cat((hidden_states_once, values_grad_once), -1)

                    z_feature = lifelong_normalizer(z_feature)
                    (hidden_states_hat, pesudo_values_hat) = z_feature.split([shape[0], shape[1]], -1)
                
                    coeffs = - self.config.lr*(hidden_states_hat * hidden_states_hat).sum(-1).unsqueeze(-1)
                v_feature = torch.cat((v_feature, coeffs * pesudo_values_hat))
            with torch.no_grad():
                mat = hidden_states.T @ hidden_states + torch.eye(shape[0], device=self.config.device)
            v_feature = v_feature[:hidden_states.shape[0], :]
            param_shift = torch.linalg.solve(mat, hidden_states.T @ v_feature)
            param_shifts[module_name] = param_shift.to(next(self.model.parameters()).device)

        return param_shifts
        

    def to(self, device):
        super().to(device)
        self.lifelong_normalizer.to(device)
        self.model.to(device)

