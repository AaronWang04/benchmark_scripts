import os
os.environ["TORCHINDUCTOR_FORCE_DISABLE_CACHES"] = "1"


import torch
import torch.nn as nn
from torch._inductor.utils import run_and_get_code, run_fw_bw_and_get_code


torch.manual_seed(0)

device = torch.device("cuda")
dtype = torch.bfloat16

class MatMulReLUModel(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(hidden_size, input_size))

    def forward(self, x):
        # MatMul + ReLU forward pass
        output = torch.mm(x, self.weight.t()) # This performs matrix multiplication
        # output = torch.relu(output)  # Apply ReLU activation
        return output

# Create model instance and input tensor
input_size = 1024
hidden_size = 1024
batch_size = 5

model = MatMulReLUModel(input_size, hidden_size).to(device=device, dtype=dtype)
input_tensor = torch.randn(batch_size, input_size, device=device, dtype=dtype, requires_grad=True)

@torch.compile(backend="inductor", options={"max-autotune": True})
def forward_pass_fn():
    return model(input_tensor)

model_output, generated_codes = run_fw_bw_and_get_code(forward_pass_fn)

print(generated_codes[0])
