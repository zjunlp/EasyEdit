import numpy as np
import torch
from typing import Union, List

class linear:
    def __init__(self):
        pass
    def execute(
        self,
        t: Union[float, List[float]],
        v0: Union[List[torch.Tensor], torch.Tensor],
        v1: Union[List[torch.Tensor], torch.Tensor],
        DOT_THRESHOLD: float = 0.9995,
        eps: float = 1e-8,
        densities = None,
    ):
        if type(v0) is list:
            v0 = v0[0]
        if type(t) is list:
            t = t[0]
        if type(v1) is list:
            v1 = v1[0]

        return t * v1 + (1.0 - t) * v0