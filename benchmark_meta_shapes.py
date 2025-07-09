import torch
import triton
from torch._inductor import config as inductor_config
from torch._inductor.utils import run_and_get_code

torch.manual_seed(0)

rtol_t = 1e-5
atol_t = 1e-5

warmup = 10
numrun = 100

# batch = 16
M = 1600
N = 60
K = 1024

strides_A = [1024, 1]
strides_B = [1, 1024]

device = torch.device("cuda")
dtype = torch.float16

A = torch.randn(M, K, device=device, dtype=dtype)
B = torch.randn(K, N, device=device, dtype=dtype)

A = A.as_strided(A.shape, strides_A)
B = B.as_strided(B.shape, strides_B)

compiled_mm = torch.compile(
    torch.mm, 
    dynamic=False, 
    options={
        "force_disable_caches": True,
        "max_autotune": True,
        "max_autotune_gemm": True,
        "max_autotune_gemm_backends": "TRITON",
        "autotune_fallback_to_aten": False,
    }
)

def func1():
    return torch.mm(A, B)

def func2():
    return compiled_mm(A, B)

result1 = func1()
result2 = func2()
torch.cuda.synchronize(device=device)

if not torch.allclose(result1, result2, rtol=rtol_t, atol=atol_t):
    diff = torch.abs(result1 - result2)
    max_diff = torch.max(diff)
    mean_diff = torch.mean(diff)
    print(f"!!! Results differ - max diff: {max_diff}, mean diff: {mean_diff}")
    

# warm up runs
for _ in range(warmup): func1()
torch.cuda.synchronize(device=device)

start_event = torch.cuda.Event(enable_timing=True)
end_event = torch.cuda.Event(enable_timing=True)

ms = triton.testing.do_bench(fn=func1, rep=100, return_mode="median")
print(f"Average Time per Iteration (Aten):\t {ms:.4f} ms")

for _ in range(warmup): func2()
torch.cuda.synchronize(device=device)

ms = triton.testing.do_bench(fn=func2, rep=100, return_mode="median")
print(f"Average Time per Iteration (Triton):\t {ms:.4f} ms")
