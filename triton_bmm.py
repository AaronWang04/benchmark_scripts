import torch

B = 0
M = 0
N = 0
K = 0
dtype = torch.float16
device = torch.device("cuda")

num_iterations = 20
warmup_iterations = 5

triton = torch.compile(torch.bmm, mode="max-autotune")
aten = torch.bmm


A_mat = torch.randn(B, K, M, dtype=dtype, device=device)
A_mat = A_mat.permute(0, 2, 1)

temp_for_B = torch.randn(N,K, dtype=dtype, device=device)
single_slice_B_data = temp_for_B.transpose(0,1) 
B_mat = torch.as_strided( single_slice_B_data, size=(B, K, N), stride=(0, 1, K))

for _ in range(warmup_iterations):
    test = aten(A_mat, B_mat)

start_event = torch.cuda.Event(enable_timing=True)
end_event = torch.cuda.Event(enable_timing=True)
start_event.record()
for _ in range(num_iterations):
    _ = aten(A_mat, B_mat)
end_event.record()
torch.cuda.synchronize()
elapsed_time_ms = start_event.elapsed_time(end_event)
avg_time_ms = elapsed_time_ms / num_iterations
print(f"Average Time per Iteration (Baseline):\t {avg_time_ms:.4f} ms")


for _ in range(warmup_iterations):
    triton(A_mat, B_mat)
start_event = torch.cuda.Event(enable_timing=True)
end_event = torch.cuda.Event(enable_timing=True)
start_event.record()
for _ in range(num_iterations):
    _ = triton(A_mat, B_mat)
end_event.record()
torch.cuda.synchronize()
elapsed_time_ms = start_event.elapsed_time(end_event)
avg_time_ms = elapsed_time_ms / num_iterations
print(f"Average Time per Iteration (Triton):\t {avg_time_ms:.4f} ms")
