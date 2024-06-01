from typing import Dict, Union, Tuple, List, Any, Literal, Optional
import torch
import numpy as np

from .utils import rescaled_random, magnitude, random_wo_rescaled


class GTA:
    def __init__(self, sparsify_method=None, consensus_method=None, normalize=False):
        self.sparsify_method = sparsify_method
        self.consensus_method = consensus_method

        self.normalize = normalize

    def execute(
            self,
            weights,
            base,
            tensors,
            densities,
            **_kwargs,
    ) -> torch.Tensor:
        # collect task vectors
        densities = [densities for _ in range(len(tensors))]
        # weights = [weights / len(tensors) for _ in range(len(tensors))]
        assert len(densities) == len(weights) == len(tensors)
        deltas, base = get_task_vectors(base, tensors)
        if not deltas:
            return base

        # sparsify
        if self.sparsify_method:
            if self.sparsify_method == 'magnitude':
                sparsify = magnitude
            elif self.sparsify_method == 'rescaled_random':
                sparsify = rescaled_random
            elif self.sparsify_method == 'random':
                sparsify = random_wo_rescaled
            else:
                raise NotImplementedError
            for i, delta in enumerate(deltas):
                deltas[i] = sparsify(
                    delta,
                    density=densities[i]
                )

        deltas = torch.stack(deltas, dim=0)
        weights = torch.tensor(
            [_ for _ in weights], dtype=deltas.dtype, device=deltas.device
        )
        while len(deltas.shape) > len(weights.shape):
            weights.unsqueeze_(-1)

        weighted_deltas = deltas * weights

        # get sign consensus and mix deltas
        if self.consensus_method:
            mask_dtype = base.dtype
            mask = get_mask(
                weighted_deltas,
                method=self.consensus_method,
                mask_dtype=mask_dtype,
            )
            mixed_delta = (weighted_deltas * mask).sum(dim=0)
            divisor = (weights * mask).sum(dim=0)
            divisor[divisor == 0] = 1
        else:
            mixed_delta = weighted_deltas.sum(dim=0)
            divisor = weights.sum(dim=0)
            divisor[divisor.abs() < 1e-8] = 1

        if self.normalize:
            mixed_delta /= divisor

        return (base + mixed_delta).to(base.dtype)

def get_task_vectors(
    base: Union[np.ndarray, torch.Tensor],
    tensors: Union[List[np.ndarray], List[torch.Tensor]],
) -> Tuple[List[Dict[str, Any]], torch.Tensor]:

    res = []
    for x in tensors:
        delta = x - base
        del x
        res.append(delta)
    return res, base

def get_mask(
    delta: torch.Tensor,
    method: Literal["sum", "count"] = "sum",
    mask_dtype: Optional[torch.dtype] = None,
):
    """Returns a mask determining which delta vectors should be merged
    into the final model.

    For the methodology described in the TIES paper use 'sum'. For a
    simpler naive count of signs, use 'count'."""
    if mask_dtype is None:
        mask_dtype = delta.dtype

    sign = delta.sign().to(mask_dtype)

    if method == "sum":
        sign_weight = delta.sum(dim=0)
        majority_sign = (sign_weight >= 0).to(mask_dtype) * 2 - 1
        del sign_weight
    elif method == "count":
        majority_sign = (sign.sum(dim=0) >= 0).to(mask_dtype) * 2 - 1
    else:
        raise RuntimeError(f'Unimplemented mask method "{method}"')

    return sign == majority_sign