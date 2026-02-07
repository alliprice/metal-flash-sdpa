"""Forward and backward timing benchmarks for Metal Flash Attention."""
import torch
import time


def bench_forward(B, H, R, C, D, dtype, n_iters=100):
    """Basic timing comparison: MFA vs default SDPA."""
    import metal_flash_sdpa

    Q = torch.randn(B, H, R, D, device="mps", dtype=dtype)
    K = torch.randn(B, H, C, D, device="mps", dtype=dtype)
    V = torch.randn(B, H, C, D, device="mps", dtype=dtype)

    # Warm up
    with torch.no_grad():
        for _ in range(10):
            _ = torch.nn.functional.scaled_dot_product_attention(Q, K, V)
    torch.mps.synchronize()

    # Time default SDPA
    metal_flash_sdpa.disable()
    start = time.time()
    with torch.no_grad():
        for _ in range(n_iters):
            _ = torch.nn.functional.scaled_dot_product_attention(Q, K, V)
    torch.mps.synchronize()
    default_time = (time.time() - start) / n_iters * 1000

    # Time MFA
    metal_flash_sdpa.enable()
    start = time.time()
    with torch.no_grad():
        for _ in range(n_iters):
            _ = torch.nn.functional.scaled_dot_product_attention(Q, K, V)
    torch.mps.synchronize()
    mfa_time = (time.time() - start) / n_iters * 1000
    metal_flash_sdpa.disable()

    speedup = default_time / mfa_time if mfa_time > 0 else float("inf")
    print(f"  Timing B={B} H={H} R={R} C={C} D={D} {dtype}: "
          f"default={default_time:.2f}ms  MFA={mfa_time:.2f}ms  speedup={speedup:.2f}x")


def bench_backward(B, H, R, C, D, dtype, n_iters=50):
    """Time forward+backward: MFA vs default SDPA."""
    import metal_flash_sdpa

    def run_fwd_bwd(Q, K, V):
        out = torch.nn.functional.scaled_dot_product_attention(Q, K, V)
        loss = out.sum()
        loss.backward()

    Q = torch.randn(B, H, R, D, device="mps", dtype=dtype, requires_grad=True)
    K = torch.randn(B, H, C, D, device="mps", dtype=dtype, requires_grad=True)
    V = torch.randn(B, H, C, D, device="mps", dtype=dtype, requires_grad=True)

    # Warm up
    for _ in range(5):
        run_fwd_bwd(Q, K, V)
        Q.grad = K.grad = V.grad = None
    torch.mps.synchronize()

    # Time default SDPA
    metal_flash_sdpa.disable()
    start = time.time()
    for _ in range(n_iters):
        run_fwd_bwd(Q, K, V)
        Q.grad = K.grad = V.grad = None
    torch.mps.synchronize()
    default_time = (time.time() - start) / n_iters * 1000

    # Time MFA
    metal_flash_sdpa.enable()
    start = time.time()
    for _ in range(n_iters):
        run_fwd_bwd(Q, K, V)
        Q.grad = K.grad = V.grad = None
    torch.mps.synchronize()
    mfa_time = (time.time() - start) / n_iters * 1000
    metal_flash_sdpa.disable()

    speedup = default_time / mfa_time if mfa_time > 0 else float("inf")
    print(f"  Fwd+Bwd B={B} H={H} R={R} C={C} D={D} {dtype}: "
          f"default={default_time:.2f}ms  MFA={mfa_time:.2f}ms  speedup={speedup:.2f}x")


if __name__ == "__main__":
    print("=== Forward Timing Benchmarks ===\n")
    bench_forward(2, 8, 128, 128, 64, torch.float16)
    bench_forward(1, 8, 512, 512, 64, torch.float16)
    bench_forward(1, 8, 2048, 2048, 64, torch.float16)

    print("\n=== Forward+Backward Timing Benchmarks ===\n")
    bench_backward(2, 8, 128, 128, 64, torch.float16)
    bench_backward(1, 8, 512, 512, 64, torch.float16)
