import torch
from typing import List, Union

def rescaled_masked_tensor(
    tensor: torch.Tensor,
    mask: torch.Tensor,
    norm: str,
    eps: float = 1e-7,
) -> torch.Tensor:
    """
    Apply mask to tensor and rescale according to specified norm.
    """
    masked = tensor * mask

    if norm is None:
        return masked

    if norm == 'l1':
        before_scale = tensor.abs().sum()
        after_scale = masked.abs().sum()
    elif norm == 'l2':
        before_scale = tensor.norm()
        after_scale = masked.norm()
    elif norm == 'linf':
        before_scale = tensor.abs().max()
        after_scale = masked.abs().max()
    else:
        raise NotImplementedError(norm)

    if before_scale < eps or after_scale < eps:
        return masked
    return masked * (before_scale / after_scale)

def magnitude(tensor, density, rescale_norm=None):
    """
    Trim a tensor by keeping only the top-k elements by magnitude.
    """
    if density >= 1:
        return tensor
    k = int(density * tensor.numel()) 

    assert k > 0, "Cannot zero out the entire tensor"
    mask = torch.zeros_like(tensor)
    w = tensor.abs().view(-1)
    if w.device.type == "cpu":
        w = w.float()
    topk = torch.argsort(w, descending=True)[:k]
    mask.view(-1)[topk] = 1

    res = rescaled_masked_tensor(tensor, mask, rescale_norm)
    return res

def bernoulli(
    tensor: torch.Tensor, 
    density: float, 
    rescale_norm: str = None
) -> torch.Tensor:
    """
    Apply random sparsification using Bernoulli distribution.
    """
    if density >= 1:
        return tensor

    if (tensor.device.type != "cpu") or tensor.dtype == torch.bfloat16:
        work_dtype = tensor.dtype
    else:
        work_dtype = torch.float32

    mask = torch.bernoulli(
        torch.full_like(input=tensor, fill_value=density, dtype=work_dtype)
    )
    res = rescaled_masked_tensor(tensor.to(work_dtype), mask, rescale_norm)
    return res.to(tensor.dtype)

def trim( 
    tensor: torch.Tensor, 
    density: float, 
    sparsification_method: str = "magnitude",
    rescale_norm: str = None
) -> torch.Tensor:
    """
    Trim tensor using specified sparsification method.
    """
    if sparsification_method == 'magnitude':
        return magnitude(tensor, density, rescale_norm)
    if sparsification_method == 'random':
        return bernoulli(tensor, density, rescale_norm)

def get_mask(
    delta: torch.Tensor,
    mask_dtype: torch.dtype,
    consensus_method: str = 'sum'
) -> torch.Tensor:
    """
    Calculate sign consensus mask for merging vectors.
    """
    if mask_dtype is None:
        mask_dtype = delta.dtype
    sign = delta.sign().to(dtype=mask_dtype)
    if consensus_method == 'sum':
        sign_weight = delta.sum(dim=0)
        majority_sign = (sign_weight >= 0).to(mask_dtype) * 2 - 1
        del sign_weight
        return sign == majority_sign

def merge_steer_vectors(
    vectors: Union[List[torch.Tensor], torch.Tensor],
    weights: Union[List[float], torch.Tensor],
    densities: Union[List[float], torch.Tensor, None] = None,
    mask_dtype: torch.dtype = torch.float32,
    sparsification_method: str = None,
    consensus_method: str = "sum",
    normalize: bool = True,
    rescale_norm: str = None
) -> torch.Tensor:
    """
    Merge multiple steering vectors using the TIES-Merging.
    """
    if isinstance(vectors, list):
        assert all(v.shape == vectors[0].shape for v in vectors), "All vectors must have the same shape"
        vectors = torch.stack(vectors, dim=0)
    
    if weights is None:
        weights = torch.ones(len(vectors), dtype=vectors.dtype, device=vectors.device)
    else:
        weights = torch.tensor(weights, dtype=vectors.dtype, device=vectors.device)
    
    if densities is None:
        densities = torch.ones(len(vectors), dtype=vectors.dtype, device=vectors.device)
    else:
        densities = torch.tensor(densities, dtype=vectors.dtype, device=vectors.device)
    
    trimmed_vectors = []
    for i in range(len(vectors)):
        if sparsification_method is not None:
            trimmed_vectors.append(trim(vectors[i], densities[i], sparsification_method, rescale_norm))
        else:
            trimmed_vectors.append(vectors[i])
    
    vectors = torch.stack(trimmed_vectors, dim=0)
    while len(weights.shape) < len(vectors.shape):
        weights.unsqueeze_(-1)
    weighted_vectors = vectors * weights
    
    if consensus_method:
        mask = get_mask(weighted_vectors, mask_dtype=mask_dtype, consensus_method=consensus_method)
        mixed_vector = (weighted_vectors * mask).sum(dim=0)
        divisor = (weights * mask).sum(dim=0)
        divisor[divisor == 0] = 1
    else:
        mixed_vector = weighted_vectors.sum(dim=0)
        divisor = weights.sum(dim=0)
        divisor[divisor.abs() < 1e-8] = 1
        
    if normalize:
        mixed_vector /= divisor

    return mixed_vector

def merge_linear(
    vectors: List[torch.Tensor],
    weights: List[float],
    normalize: bool = True
) -> torch.Tensor:
    """
    Merge vectors using linear combination method.
    """
    stacked_vectors = torch.stack(vectors, dim=0)
    tensor_weights = torch.tensor(weights, dtype=stacked_vectors.dtype, device=stacked_vectors.device)
    
    while len(tensor_weights.shape) < len(stacked_vectors.shape):
        tensor_weights.unsqueeze_(-1)
        
    merged_vector = (tensor_weights * stacked_vectors).sum(dim=0)
    
    if normalize:
        merged_vector = merged_vector / tensor_weights.sum(dim=0)
        
    return merged_vector
