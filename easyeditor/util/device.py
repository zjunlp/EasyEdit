from collections.abc import Mapping

import torch


def normalize_device(device=None, *, default_cuda=True):
    if isinstance(device, torch.device):
        return device

    if device is None:
        if default_cuda and torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    if isinstance(device, int):
        return torch.device(f"cuda:{device}")

    device_str = str(device).strip()
    if device_str == "":
        return normalize_device(None, default_cuda=default_cuda)

    if device_str.isdigit():
        return torch.device(f"cuda:{device_str}")

    return torch.device(device_str)


def move_to_device(obj, device):
    target_device = normalize_device(device)

    if hasattr(obj, "to"):
        return obj.to(target_device)

    if isinstance(obj, Mapping):
        return type(obj)((key, move_to_device(value, target_device)) for key, value in obj.items())

    if isinstance(obj, tuple):
        return tuple(move_to_device(item, target_device) for item in obj)

    if isinstance(obj, list):
        return [move_to_device(item, target_device) for item in obj]

    return obj


def get_model_device(model, fallback=None):
    if model is not None and hasattr(model, "device"):
        return normalize_device(model.device)

    if model is not None:
        try:
            return next(model.parameters()).device
        except (AttributeError, StopIteration):
            pass

    return normalize_device(fallback)


def get_module_device(module, fallback=None):
    if module is not None and hasattr(module, "device"):
        return normalize_device(module.device)

    if module is not None:
        try:
            return next(module.parameters()).device
        except (AttributeError, StopIteration):
            pass

        try:
            return next(module.buffers()).device
        except (AttributeError, StopIteration):
            pass

    return normalize_device(fallback)


def copy_to_param(param, value):
    with torch.no_grad():
        if not torch.is_tensor(value):
            value = torch.as_tensor(value)
        param[...] = value.to(device=param.device, dtype=param.dtype)
