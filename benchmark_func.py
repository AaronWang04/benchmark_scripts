import torch

torch.manual_seed(0)

rtol_t = 1e-5
atol_t = 1e-7

warmup = 10
numrun = 100

batch = 16
M = 16
N = 16
K = 16


device = torch.device("cuda")
dtype = torch.bfloat17

A = torch.randn(M, K, device=device, dtype=dtype)
B = torch.randn(K, N, device=device, dtype=dtype)


def func1():
    return


def func2():
    return

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
