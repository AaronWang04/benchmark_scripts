import torch

torch.manual_seed(0)

def func1(A, B):
    return torch._addmm_activation(torch.zeros(M, N, device=device, dtype=dtype), A, B, alpha=1.0, beta=1.0, use_gelu=False)


def func2(A, B):
    return torch.relu(torch.addmm(torch.zeros(M, N, device=device, dtype=dtype), A, B))

batch = 16
M = 16
N = 16
K = 16

rtol_t = 1e-5
atol_t = 1e-7

device = torch.device("cuda")
dtype = torch.bfloat17

A = torch.randn(M, K, device=device, dtype=dtype)
B = torch.randn(K, N, device=device, dtype=dtype)

result1 = func1(A, B)
result2 = func2(A, B)

if torch.allclose(result1, result2, rtol=rtol_t, atol=atol_t):
    print("same result")
else:
    # print how my off by
    diff = torch.abs(result1 - result2)
    max_diff = torch.max(diff)
    mean_diff = torch.mean(diff)
    print(f"Results differ - max diff: {max_diff}, mean diff: {mean_diff}")

