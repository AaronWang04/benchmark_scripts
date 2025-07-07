import os
import torch

os.environ["TORCHINDUCTOR_FORCE_DISABLE_CACHES"] = "1"

dtype = torch.float16
device = torch.device("cuda")

rtol_t = 1e-5
atol_t = 1e-7

warmup = 10
numrun = 100

M = 1600
N = 60
K = 1024

strideA = [1024,1]
strideB = [1, 1024]
strideC = [60, 1]

A = torch.randn(M, K, dtype=dtype, device=device)
A = A.as_strided((M, K), strideA)
B = torch.randn(K, N, dtype=dtype, device=device)
B = B.as_strided((K, N), strideB)
# C = A @ B

def func1():
    return A @ B

def func2():
    compiled_func = torch.compile(lambda: A @ B, mode="max-autotune")
    return compiled_func()

result1 = func1()
result2 = func2()

if torch.allclose(result1, result2, rtol=rtol_t, atol=atol_t):
    print("same result")
else:
    # print how my off by
    diff = torch.abs(result1 - result2)
    max_diff = torch.max(diff)
    mean_diff = torch.mean(diff)
    print(f"Results differ - max diff: {max_diff}, mean diff: {mean_diff}")

# warm up runs
for _ in range(warmup): func1()
torch.cuda.synchronize(device=device)

start_event = torch.cuda.Event(enable_timing=True)
end_event = torch.cuda.Event(enable_timing=True)

total_time_ms = 0.0
start_event.record()
for _ in range(numrun): func1()
end_event.record()
torch.cuda.synchronize(device=device)
total_time_ms += start_event.elapsed_time(end_event)
avg_time_ms = total_time_ms / numrun

print(f"Average Time per Iteration (Func 1):\t {avg_time_ms:.4f} ms")

for _ in range(warmup): func2()
torch.cuda.synchronize(device=device)

start_event = torch.cuda.Event(enable_timing=True)
end_event = torch.cuda.Event(enable_timing=True)

total_time_ms = 0.0
start_event.record()
for _ in range(numrun): func2()
end_event.record()
torch.cuda.synchronize(device=device)
total_time_ms += start_event.elapsed_time(end_event)
avg_time_ms = total_time_ms / numrun

print(f"Average Time per Iteration (Func 2):\t {avg_time_ms:.4f} ms")

