import torch
import triton
from torch._inductor import config as inductor_config


# M, N, K
SHAPES = [
    (1600, 128, 378, "[378, 1], [1, 378], [128, 1]"),
    (200, 3552, 1024, "[1024, 1], [1, 1024], [3552, 1]"),
    (1600, 192, 512, "[512, 1], [1, 512], [192, 1]"),
    (200, 1024, 2368, "[2368, 1], [1, 2368], [1024, 1]"),
    (200, 2048, 1024, "[1024, 1], [1, 1024], [2048, 1]"),
    (1600, 1024, 1024, "[1024, 1], [1, 1024], [1024, 1]"),
    (200, 6144, 1024, "[1024, 1], [1, 1024], [6144, 1]"),
    (1600, 378, 128, "[128, 1], [1, 128], [378, 1]"),
    (1600, 144, 1024, "[1024, 1], [1, 1024], [144, 1]"),
    (1600, 160, 1024, "[1024, 1], [1, 1024], [160, 1]"),
    (1600, 60, 1024, "[1024, 1], [1, 1024], [60, 1]"),
    (1600, 256, 256, "[256, 1], [1, 256], [256, 1]"),
    (1600, 2560, 256, "[256, 1], [1, 256], [2560, 1]"),
    (1600, 1024, 64, "[64, 1], [1, 64], [1024, 1]"),
    (1600, 3328, 512, "[512, 1], [1, 512], [3328, 1]"),
    (1600, 256, 1024, "[1024, 1], [1, 1024], [256, 1]"),
    (1600, 512, 512, "[512, 1], [1, 512], [512, 1]"),
    (1600, 2, 32, "[32, 1], [1, 32], [2, 1]"),
    (1600, 256, 128, "[128, 1], [1, 128], [256, 1]"),
    (1600, 256, 160, "[160, 1], [1, 160], [256, 1]"),
]


def parse_strides_str(strides_str):
    """Parse strides string into a list of strides for A, B, and C tensors."""
    return [
        list(map(int, stride.strip().split(",")))
        for stride in strides_str.strip("[]").split("], [")
    ]


def create_tensors_with_strides(m, n, k, strides_str):
    """Create tensors with appropriate shapes and strides based on the provided parameters."""
    device = "cuda"
    dtype = torch.float16

    # Parse strides for A, B, and C tensors
    strides = parse_strides_str(strides_str)
    a_strides, b_strides, c_strides = strides[:3]

    # Create contiguous tensors first
    a_contig = torch.randn((m, k), device=device, dtype=dtype)
    b_contig = torch.randn((k, n), device=device, dtype=dtype)

    # Create tensors with the specified strides
    a = a_contig.as_strided((m, k), a_strides)
    b = b_contig.as_strided((k, n), b_strides)

    return a, b


def aten_matmul(a, b):
    return lambda: torch.matmul(a, b)


def pt2_triton_matmul(a, b):
    torch._dynamo.reset()
    with inductor_config.patch(
        force_disable_caches=True,
        max_autotune=True,
        max_autotune_gemm=True,
        max_autotune_gemm_backends="TRITON",
        autotune_fallback_to_aten=False,
    ):
        f = lambda a, b: a.matmul(b)
        compiled = torch.compile(f, dynamic=False)
        compiled(a, b)
    return lambda: compiled(a, b)


def main():
    performance_times = []
    shapes = SHAPES

    # Define kernels to benchmark
    KERNELS = [
        aten_matmul,
        pt2_triton_matmul,
    ]

    # Process each shape
    for shape in shapes:
        print(f"processing {shape=}")
        m, n, k, strides_str = shape
        print(f"  dimensions: m={m}, n={n}, k={k}")
        print(f"  strides: {strides_str}")

        # Create tensors with appropriate strides
        a, b = create_tensors_with_strides(m, n, k, strides_str)
        ref_output = aten_matmul(a, b)()
        one_performance_times = []

        for kernel in KERNELS:
            print(f"kernel: {kernel.__name__}", flush=True)

            # Run the benchmark
            fn = kernel(a, b)
            actual_output = fn()

            # Check accuracy
            if not torch.allclose(ref_output, actual_output, rtol=1e-03, atol=1e-03):
                print(
                    f"!!!!!ACCURACY WARNING: accuracy test failed: {kernel=} with {shape=}",
                    flush=True,
                )

            ms = triton.testing.do_bench(fn=fn, rep=100, return_mode="median")
            # ms = triton.testing.do_bench_cudagraph(fn=fn, rep=100, return_mode="median")
            one_performance_times.append(ms)

        performance_times.append((shape, one_performance_times))
        print(
            f"performance_times for {shape=}: "
            + ",".join([str(t) for t in one_performance_times]),
            flush=True,
        )

    print("performance_times:", flush=True)
    print("shape," + ",".join([k.__name__ for k in KERNELS]))
    for shape, times in performance_times:
        print(
            f"{shape},"
            + ",".join([f"{t:.8f}" for t in times]),
            "\ttriton faster" if times[1] < times[0] else "aten faster",
            flush=True,
        )


if __name__ == "__main__":
    main()