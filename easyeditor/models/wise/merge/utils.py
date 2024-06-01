import torch

def magnitude(tensor: torch.Tensor, density: float) -> torch.Tensor:
    """Masks out the smallest values, retaining a proportion of `density`."""
    if density >= 1:
        return tensor

    k = int(density * tensor.view(-1).shape[0])

    assert k > 0, "not gonna zero out the whole tensor buddy"
    mask = torch.zeros_like(tensor)
    w = tensor.abs().view(-1)
    if w.device.type == "cpu":
        w = w.float()
    topk = torch.topk(w, k=k, largest=True)
    mask.view(-1)[topk.indices] = 1

    return tensor * mask


def bernoulli(
    tensor: torch.Tensor, density: float, rescale: bool = True
) -> torch.Tensor:
    if density >= 1:
        return tensor

    if (tensor.device.type != "cpu") or tensor.dtype == torch.bfloat16:
        work_dtype = tensor.dtype
    else:
        # torch.bernoulli not implemented for float16 on CPU, upcast to float32
        work_dtype = torch.float32

    mask = torch.bernoulli(
        torch.full_like(input=tensor, fill_value=density, dtype=work_dtype)
    )
    res = tensor.to(work_dtype) * mask
    if rescale:
        res /= density
    return res.to(tensor.dtype)

def rescaled_random(tensor: torch.Tensor, density: float):
    return bernoulli(tensor, density, rescale=True)

def random_wo_rescaled(tensor: torch.Tensor, density: float):
    return bernoulli(tensor, density, rescale=False)