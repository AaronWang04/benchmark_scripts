import torch
import torch.nn as nn
from torch.autograd.functional import jacobian

torch.manual_seed(0)

device = torch.device("cuda")

normalized_shape_arg = (3, 3, 3)
input_tensor = torch.randn(3, 3, 3, device=device, requires_grad=True)
weight_tensor = torch.randn(3,3,3 ,device=device, requires_grad=True)
output_tensor = torch.nn.functional.rms_norm(input_tensor, normalized_shape_arg, weight_tensor)

def func(x, w):
    return torch.rms_norm(x, normalized_shape_arg, w)

strategy = "forward-mode"
jacobian_x = jacobian(func, (input_tensor, weight_tensor), vectorize=True, strategy=strategy)[0]
jacobian_weight = jacobian(func, (input_tensor, weight_tensor), vectorize=True, strategy=strategy)[1]
print(torch.flatten(jacobian_weight)[:50])

