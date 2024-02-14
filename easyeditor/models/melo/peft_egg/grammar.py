# import torch
#
# lora_A = torch.randn((6,1))
# tensor_list = [torch.tensor(float(i)) for i in range(1,5)]
# print(f'lora_A: {lora_A}')
# print(f'tensor_list: {tensor_list}')
#
# lora_A.requires_grad = True
# for x in tensor_list:
#     x.requires_grad = True
#
# c = []
# for x in tensor_list:
#     c.append(lora_A[1] * x)
#
# d = torch.stack(c,0)
# print(f'stacked d: {d}')
#
# d.sum().backward()
#
# print(lora_A.grad)


import torch
import torch.nn as nn
class Conv1D(nn.Module):
    """
    1D-convolutional layer as defined by Radford et al. for OpenAI GPT (and also used in GPT-2).

    Basically works like a linear layer but the weights are transposed.

    Args:
        nf (`int`): The number of output features.
        nx (`int`): The number of input features.
    """

    def __init__(self, nf, nx):
        super().__init__()
        self.nf = nf
        self.weight = nn.Parameter(torch.empty(nx, nf))
        self.bias = nn.Parameter(torch.zeros(nf))
        nn.init.normal_(self.weight, std=0.02)

    def forward(self, x):
        size_out = x.size()[:-1] + (self.nf,)
        x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        x = x.view(size_out)
        return x
a = Conv1D(500,300)

print(a.weight.shape)