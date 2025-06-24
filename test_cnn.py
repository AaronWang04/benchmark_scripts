import torch
import torch.nn as nn

device = torch.device("cuda")
print(torch.backends.cudnn.is_available())
print(torch.backends.cudnn.enabled)

def test_conv_performance(batch_size, in_channels, out_channels, input_size, kernel_size, stride, padding, dtype, num_runs=100, warmup_runs=10, memory_format=torch.contiguous_format):

    model = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding).to(device, dtype=dtype).to(memory_format=memory_format)
    input_tensor = torch.randn(batch_size, in_channels, input_size, input_size, device=device, dtype=dtype).to(memory_format=memory_format)

    # Warm-up GPU: execute the operation a few times before timing
    for _ in range(warmup_runs):
        _ = model(input_tensor)

    torch.cuda.synchronize(device=device) # Wait for all CUDA cores to finish warmup operations

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    total_time_ms = 0.0
    start_event.record()
    for _ in range(num_runs):
        _ = model(input_tensor)
    end_event.record()

    torch.cuda.synchronize(device=device)
    total_time_ms += start_event.elapsed_time(end_event)
    avg_time_ms = total_time_ms / num_runs
    return avg_time_ms

if __name__ == "__main__":
    batch_size = 16
    in_channels = 3
    out_channels = 64
    input_size = 224
    kernel_size = 3
    stride = 1
    padding = 1
    num_runs = 20
    warmup_runs = 5
    memory_format=torch.contiguous_format
    # memory_format=torch.channels_last

    results = {}

    # Test FP32 (Single-precision floating-point)
    print("Testing FP32...")
    fp32_time = test_conv_performance(batch_size, in_channels, out_channels, input_size, kernel_size, stride, padding, torch.float32, num_runs, warmup_runs, memory_format)
    print(f"FP32 Average Time: {fp32_time:.3f} ms")
    results['fp32'] = fp32_time
    print("-" * 30)

    # Test FP16 (Half-precision floating-point)
    print("Testing FP16...")
    fp16_time = test_conv_performance(batch_size, in_channels, out_channels, input_size, kernel_size, stride, padding, torch.float16, num_runs, warmup_runs, memory_format)
    print(f"FP16 Average Time: {fp16_time:.3f} ms")
    results['fp16'] = fp16_time
    if results.get('fp32'):
        print(f"  Speedup vs FP32: {results['fp32'] / fp16_time:.2f}x")
    print("-" * 30)

    print("Testing BF16...")
    bf16_time = test_conv_performance(batch_size, in_channels, out_channels, input_size, kernel_size, stride, padding, torch.bfloat16, num_runs, warmup_runs, memory_format)
    print(f"BF16 Average Time: {bf16_time:.3f} ms")
    results['bf16'] = bf16_time
    if results.get('fp32'):
        print(f"  Speedup vs FP32: {results['fp32'] / bf16_time:.2f}x")

    print("-" * 50)
    print("Testing complete.")