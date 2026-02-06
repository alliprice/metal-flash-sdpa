"""Validate Metal Flash Attention forward pass against PyTorch eager attention."""
import torch
import time


def reference_sdpa(Q, K, V, scale):
    """Eager attention on CPU as reference."""
    # Q: [B, H, R, D], K: [B, H, C, D], V: [B, H, C, D]
    attn = torch.matmul(Q, K.transpose(-2, -1)) * scale
    attn = torch.softmax(attn, dim=-1)
    return torch.matmul(attn, V)


def test_config(B, H, R, C, D, dtype, Hk=None):
    """Test a single configuration."""
    if Hk is None:
        Hk = H

    Q = torch.randn(B, H, R, D, device='mps', dtype=dtype)
    K = torch.randn(B, Hk, C, D, device='mps', dtype=dtype)
    V = torch.randn(B, Hk, C, D, device='mps', dtype=dtype)

    scale = D ** -0.5

    # MFA path
    import metal_flash_sdpa
    metal_flash_sdpa.enable()
    with torch.no_grad():
        out_mfa = torch.nn.functional.scaled_dot_product_attention(Q, K, V, scale=scale)
    metal_flash_sdpa.disable()

    # Reference: eager math on CPU in float32
    out_ref = reference_sdpa(
        Q.cpu().float(), K.cpu().float(), V.cpu().float(), scale
    ).to(dtype).to('mps')

    max_diff = (out_mfa - out_ref).abs().max().item()
    mean_diff = (out_mfa - out_ref).abs().mean().item()

    # Tolerance depends on dtype
    if dtype == torch.float32:
        tol = 1e-3
    else:
        tol = 0.05  # fp16/bf16 have larger numerical differences

    status = "PASS" if max_diff < tol else "FAIL"
    print(f"  [{status}] B={B} H={H} Hk={Hk} R={R} C={C} D={D} {dtype}: "
          f"max_diff={max_diff:.6f} mean_diff={mean_diff:.6f}")
    assert max_diff < tol, f"Output mismatch! max_diff={max_diff} > tol={tol}"
    return max_diff


def test_timing(B, H, R, C, D, dtype, n_iters=100):
    """Basic timing comparison."""
    import metal_flash_sdpa

    Q = torch.randn(B, H, R, D, device='mps', dtype=dtype)
    K = torch.randn(B, H, C, D, device='mps', dtype=dtype)
    V = torch.randn(B, H, C, D, device='mps', dtype=dtype)

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

    speedup = default_time / mfa_time if mfa_time > 0 else float('inf')
    print(f"  Timing B={B} H={H} R={R} C={C} D={D} {dtype}: "
          f"default={default_time:.2f}ms  MFA={mfa_time:.2f}ms  speedup={speedup:.2f}x")


if __name__ == '__main__':
    print("=== Metal Flash Attention Forward Pass Tests ===\n")

    print("Correctness tests:")
    # Basic configs
    test_config(1, 1, 64, 64, 64, torch.float16)
    test_config(2, 8, 128, 128, 64, torch.float16)
    test_config(1, 4, 256, 256, 128, torch.float16)
    test_config(4, 8, 512, 512, 64, torch.float16)

    # float32
    test_config(2, 4, 128, 128, 64, torch.float32)

    # Different R and C
    test_config(1, 4, 64, 256, 64, torch.float16)
    test_config(1, 4, 256, 64, 64, torch.float16)

    # Large sequence
    test_config(1, 8, 2048, 2048, 64, torch.float16)

    print("\nTiming tests:")
    test_timing(2, 8, 128, 128, 64, torch.float16)
    test_timing(1, 8, 512, 512, 64, torch.float16)
    test_timing(1, 8, 2048, 2048, 64, torch.float16)

    print("\n=== All tests passed! ===")
