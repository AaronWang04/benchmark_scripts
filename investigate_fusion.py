import os
os.environ["TORCHINDUCTOR_FORCE_DISABLE_CACHES"] = "1"


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch._inductor.utils import run_and_get_code


torch.manual_seed(0)

device = torch.device("cuda")
dtype = torch.bfloat16

class MatMulReLUModel(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(input_size, input_size))
        self.bias = nn.Parameter(torch.randn(input_size, input_size))

    def forward(self, x):
        output = F.gelu(torch.addmm(self.bias, x, self.weight), approximate="tanh")
        # output = F.relu(torch.addmm(self.bias, x, self.weight))
        return output

# Create model instance and input tensor
input_size = 1024

model = MatMulReLUModel(input_size).to(device=device, dtype=dtype)
input_tensor = torch.randn(input_size, input_size, device=device, dtype=dtype, requires_grad=True)

@torch.compile(backend="inductor")
def forward_pass_fn():
    return model(input_tensor)

with torch.inference_mode():
    model_output, generated_codes = run_and_get_code(forward_pass_fn)
