import torch
import torch.nn as nn
from torch._inductor.utils import run_and_get_code, run_fw_bw_and_get_code

torch.manual_seed(0)

device = torch.device("cuda")

normalized_shape_arg = (3, 3, 3)
input_tensor = torch.randn(3, 3, 3, device=device, requires_grad=True)
weight_tensor = torch.randn(3, 3, 3 ,device=device, requires_grad=True)
output_tensor = torch.nn.functional.rms_norm(input_tensor, normalized_shape_arg, weight_tensor)

@torch.compile
def rms_norm_sinh(input_tensor, normalized_shape_arg, weight_tensor):
    output = torch.nn.functional.rms_norm(input_tensor, normalized_shape_arg, weight_tensor)
    return torch.sinh(output)

def forward_pass_fn():
    return rms_norm_sinh(input_tensor, normalized_shape_arg, weight_tensor)

model_output, generated_codes = run_fw_bw_and_get_code(forward_pass_fn)

print(generated_codes[1])