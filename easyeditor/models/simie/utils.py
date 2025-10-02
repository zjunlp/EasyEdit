from typing import Union, Tuple, List, Dict

import torch
import torch.nn as nn
from transformers.pytorch_utils import Conv1D

from ...util import nethook

def get_module(module: nn.Module, module_name: str) -> nn.Module:
    
    for name in module_name.split("."):
        module = getattr(module, name)
    return module

def get_shape(module: Union[nn.Linear, Conv1D]) -> Tuple[int]:
    
    shape = tuple(module.weight.shape)
    return shape[::-1] if isinstance(module, nn.Linear) else shape
    
def get_parameter(model, name):
    """
    Finds the named parameter within the given model.
    """
    for n, p in model.named_parameters():
        if n == name:
            return p
    raise LookupError(name)

def set_parameter(model, weights_copy, device):
    """
    Sets the named parameter within the given model.
    """
    with torch.no_grad():
        for k, v in weights_copy.items():
            nethook.get_parameter(model, k)[...] = v.to(device)
    return model

class Tracer:

    def __init__(
        self,
        module: nn.Module,
        cache_mask: torch.LongTensor
    ):
        cache_indices = torch.where(cache_mask)

        def forward_hook(
            module: nn.Module,
            inputs: Tuple[torch.FloatTensor],
            outputs: Tuple[torch.FloatTensor]
        ):
            self.keys = inputs[0][cache_indices].detach()
            self.values = outputs[cache_indices].detach()

        self.handles = [
            module.register_forward_hook(forward_hook),
        ]


class TracerDict(dict):
    
    def __init__(
        self,
        model: nn.Module,
        config,
        tuples: Dict[str, torch.LongTensor]
    ):
        
        if any("encoder" in m for m in config["inner_params"]) and any("decoder" in m for m in config.model.edit_modules):
            
            for module_name in config["inner_params"]:
                if "encoder" in module_name:
                    cache_mask = tuples["attention_mask"]
                else:
                    cache_mask = tuples["decoder_attention_mask"]
                module = get_module(model, module_name)
                self[module_name] = Tracer(module, cache_mask)

        else:

            if config["token"] == "ans":
                cache_mask = tuples["labels"] != -100
            else:
                cache_mask = tuples["attention_mask"]

            for module_name in config["inner_params"]:
                module = get_module(model, module_name)
                self[module_name] = Tracer(module, cache_mask)
            
    def __enter__(self):
        return self
            
    def __exit__(self, type, value, traceback):
        for v in self.values():
            for h in v.handles:
                h.remove()