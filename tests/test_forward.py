"""Validate Metal Flash Attention forward and backward passes against PyTorch eager attention."""
import torch
import time


def reference_sdpa(Q, K, V, scale):
    """Eager attention on CPU as reference."""
    # Q: [B, H, R, D], K: [B, H, C, D], V: [B, H, C, D]
    attn = torch.matmul(Q, K.transpose(-2, -1)) * scale
    attn = torch.softmax(attn, dim=-1)
    return torch.matmul(attn, V)


def test_config(B, H, R, C, D, dtype, Hk=None):
    """Test a single forward configuration."""
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


def test_backward(B, H, R, C, D, dtype):
    """Test backward pass: compare MFA dQ/dK/dV against CPU eager reference."""
    import metal_flash_sdpa

    scale = D ** -0.5

    # Create reference tensors on CPU in float32 for numerical stability
    Q_cpu = torch.randn(B, H, R, D, dtype=torch.float32, requires_grad=True)
    K_cpu = torch.randn(B, H, C, D, dtype=torch.float32, requires_grad=True)
    V_cpu = torch.randn(B, H, C, D, dtype=torch.float32, requires_grad=True)

    # CPU reference forward + backward
    out_ref = reference_sdpa(Q_cpu, K_cpu, V_cpu, scale)
    grad_out_cpu = torch.randn_like(out_ref)
    out_ref.backward(grad_out_cpu)
    dQ_ref = Q_cpu.grad.clone()
    dK_ref = K_cpu.grad.clone()
    dV_ref = V_cpu.grad.clone()

    # MFA forward + backward on MPS
    Q_mps = Q_cpu.detach().to(device='mps', dtype=dtype).requires_grad_(True)
    K_mps = K_cpu.detach().to(device='mps', dtype=dtype).requires_grad_(True)
    V_mps = V_cpu.detach().to(device='mps', dtype=dtype).requires_grad_(True)
    grad_out_mps = grad_out_cpu.to(device='mps', dtype=dtype)

    metal_flash_sdpa.enable()
    out_mfa = torch.nn.functional.scaled_dot_product_attention(Q_mps, K_mps, V_mps, scale=scale)
    out_mfa.backward(grad_out_mps)
    metal_flash_sdpa.disable()

    dQ_mfa = Q_mps.grad.cpu().float()
    dK_mfa = K_mps.grad.cpu().float()
    dV_mfa = V_mps.grad.cpu().float()

    # Compare
    dQ_diff = (dQ_mfa - dQ_ref).abs().max().item()
    dK_diff = (dK_mfa - dK_ref).abs().max().item()
    dV_diff = (dV_mfa - dV_ref).abs().max().item()

    if dtype == torch.float32:
        tol = 1e-3
    else:
        tol = 0.1  # fp16 backward has more accumulated error

    status = "PASS" if max(dQ_diff, dK_diff, dV_diff) < tol else "FAIL"
    print(f"  [{status}] B={B} H={H} R={R} C={C} D={D} {dtype}: "
          f"dQ_max={dQ_diff:.6f} dK_max={dK_diff:.6f} dV_max={dV_diff:.6f}")
    assert dQ_diff < tol, f"dQ mismatch! max_diff={dQ_diff} > tol={tol}"
    assert dK_diff < tol, f"dK mismatch! max_diff={dK_diff} > tol={tol}"
    assert dV_diff < tol, f"dV mismatch! max_diff={dV_diff} > tol={tol}"


def test_grad_flow():
    """Smoke test: ensure gradients flow through the patched SDPA."""
    import metal_flash_sdpa

    B, H, R, D = 2, 4, 64, 64
    Q = torch.randn(B, H, R, D, device='mps', dtype=torch.float32, requires_grad=True)
    K = torch.randn(B, H, R, D, device='mps', dtype=torch.float32, requires_grad=True)
    V = torch.randn(B, H, R, D, device='mps', dtype=torch.float32, requires_grad=True)

    metal_flash_sdpa.enable()
    out = torch.nn.functional.scaled_dot_product_attention(Q, K, V)
    loss = out.sum()
    loss.backward()
    metal_flash_sdpa.disable()

    assert Q.grad is not None, "Q.grad is None"
    assert K.grad is not None, "K.grad is None"
    assert V.grad is not None, "V.grad is None"
    assert Q.grad.abs().sum() > 0, "Q.grad is all zeros"
    assert K.grad.abs().sum() > 0, "K.grad is all zeros"
    assert V.grad.abs().sum() > 0, "V.grad is all zeros"
    print("  [PASS] Gradient flow smoke test")


def test_training_loop():
    """Smoke test: 10-step training loop with loss decreasing."""
    import metal_flash_sdpa

    B, H, R, D = 1, 4, 32, 64
    target = torch.randn(B, H, R, D, device='mps', dtype=torch.float32)

    # Simple model: project -> attention -> project
    proj_in = torch.randn(D, D, device='mps', dtype=torch.float32, requires_grad=True)
    proj_out = torch.randn(D, D, device='mps', dtype=torch.float32, requires_grad=True)

    metal_flash_sdpa.enable()
    losses = []
    lr = 0.001
    for step in range(10):
        x = torch.randn(B, H, R, D, device='mps', dtype=torch.float32)
        q = x @ proj_in
        k = x @ proj_in
        v = x @ proj_in
        out = torch.nn.functional.scaled_dot_product_attention(q, k, v)
        out = out @ proj_out
        loss = ((out - target) ** 2).mean()
        loss.backward()
        losses.append(loss.item())
        with torch.no_grad():
            proj_in -= lr * proj_in.grad
            proj_out -= lr * proj_out.grad
            proj_in.grad.zero_()
            proj_out.grad.zero_()
    metal_flash_sdpa.disable()

    # Loss should generally decrease (not strictly, but last < first)
    print(f"  Training losses: {[f'{l:.4f}' for l in losses]}")
    status = "PASS" if losses[-1] < losses[0] else "WARN"
    print(f"  [{status}] Training loop: loss {losses[0]:.4f} -> {losses[-1]:.4f}")


def test_min_seq_len_threshold():
    """Verify that the MIN_SEQ_LEN threshold routes small sequences to original SDPA."""
    import metal_flash_sdpa
    from unittest.mock import patch

    D = 64
    scale = D ** -0.5

    metal_flash_sdpa.enable()

    # Small sequence: should NOT call MFA
    Q_small = torch.randn(1, 4, 64, D, device='mps', dtype=torch.float16)
    K_small = torch.randn(1, 4, 64, D, device='mps', dtype=torch.float16)
    V_small = torch.randn(1, 4, 64, D, device='mps', dtype=torch.float16)

    with patch('metal_flash_sdpa.mfa_attention_forward') as mock_fwd:
        with torch.no_grad():
            torch.nn.functional.scaled_dot_product_attention(Q_small, K_small, V_small)
        assert not mock_fwd.called, "MFA should NOT be called for small sequences"

    # Large sequence: SHOULD call MFA
    Q_large = torch.randn(1, 4, 512, D, device='mps', dtype=torch.float16)
    K_large = torch.randn(1, 4, 512, D, device='mps', dtype=torch.float16)
    V_large = torch.randn(1, 4, 512, D, device='mps', dtype=torch.float16)

    with patch('metal_flash_sdpa.mfa_attention_forward', wraps=metal_flash_sdpa.mfa_attention_forward) as mock_fwd:
        with torch.no_grad():
            torch.nn.functional.scaled_dot_product_attention(Q_large, K_large, V_large)
        assert mock_fwd.called, "MFA should be called for large sequences"

    metal_flash_sdpa.disable()
    print(f"  [PASS] MIN_SEQ_LEN threshold (threshold={metal_flash_sdpa.MIN_SEQ_LEN})")


def reference_causal_sdpa(Q, K, V, scale):
    """Eager causal attention on CPU as reference."""
    # Q: [B, H, R, D], K: [B, H, C, D], V: [B, H, C, D]
    attn = torch.matmul(Q, K.transpose(-2, -1)) * scale
    R, C = attn.shape[-2], attn.shape[-1]
    # Create causal mask: upper triangle (future tokens) = -inf
    causal_mask = torch.triu(torch.ones(R, C, dtype=torch.bool), diagonal=1)
    attn = attn.masked_fill(causal_mask, float('-inf'))
    attn = torch.softmax(attn, dim=-1)
    return torch.matmul(attn, V)


def test_causal_forward(B, H, R, D, dtype):
    """Test causal forward: MFA output vs CPU causal reference."""
    import metal_flash_sdpa

    Q = torch.randn(B, H, R, D, device='mps', dtype=dtype)
    K = torch.randn(B, H, R, D, device='mps', dtype=dtype)
    V = torch.randn(B, H, R, D, device='mps', dtype=dtype)

    scale = D ** -0.5

    # MFA causal path
    metal_flash_sdpa.enable()
    with torch.no_grad():
        out_mfa = torch.nn.functional.scaled_dot_product_attention(
            Q, K, V, scale=scale, is_causal=True)
    metal_flash_sdpa.disable()

    # Reference: eager causal on CPU in float32
    out_ref = reference_causal_sdpa(
        Q.cpu().float(), K.cpu().float(), V.cpu().float(), scale
    ).to(dtype).to('mps')

    max_diff = (out_mfa - out_ref).abs().max().item()
    mean_diff = (out_mfa - out_ref).abs().mean().item()

    if dtype == torch.float32:
        tol = 1e-3
    else:
        tol = 0.05

    status = "PASS" if max_diff < tol else "FAIL"
    print(f"  [{status}] Causal fwd B={B} H={H} R={R} D={D} {dtype}: "
          f"max_diff={max_diff:.6f} mean_diff={mean_diff:.6f}")
    assert max_diff < tol, f"Causal forward mismatch! max_diff={max_diff} > tol={tol}"
    return max_diff


def test_causal_backward(B, H, R, D, dtype):
    """Test causal backward: dQ/dK/dV vs CPU causal reference."""
    import metal_flash_sdpa

    scale = D ** -0.5

    # CPU reference
    Q_cpu = torch.randn(B, H, R, D, dtype=torch.float32, requires_grad=True)
    K_cpu = torch.randn(B, H, R, D, dtype=torch.float32, requires_grad=True)
    V_cpu = torch.randn(B, H, R, D, dtype=torch.float32, requires_grad=True)

    out_ref = reference_causal_sdpa(Q_cpu, K_cpu, V_cpu, scale)
    grad_out_cpu = torch.randn_like(out_ref)
    out_ref.backward(grad_out_cpu)
    dQ_ref = Q_cpu.grad.clone()
    dK_ref = K_cpu.grad.clone()
    dV_ref = V_cpu.grad.clone()

    # MFA causal on MPS
    Q_mps = Q_cpu.detach().to(device='mps', dtype=dtype).requires_grad_(True)
    K_mps = K_cpu.detach().to(device='mps', dtype=dtype).requires_grad_(True)
    V_mps = V_cpu.detach().to(device='mps', dtype=dtype).requires_grad_(True)
    grad_out_mps = grad_out_cpu.to(device='mps', dtype=dtype)

    metal_flash_sdpa.enable()
    out_mfa = torch.nn.functional.scaled_dot_product_attention(
        Q_mps, K_mps, V_mps, scale=scale, is_causal=True)
    out_mfa.backward(grad_out_mps)
    metal_flash_sdpa.disable()

    dQ_mfa = Q_mps.grad.cpu().float()
    dK_mfa = K_mps.grad.cpu().float()
    dV_mfa = V_mps.grad.cpu().float()

    dQ_diff = (dQ_mfa - dQ_ref).abs().max().item()
    dK_diff = (dK_mfa - dK_ref).abs().max().item()
    dV_diff = (dV_mfa - dV_ref).abs().max().item()

    if dtype == torch.float32:
        tol = 1e-3
    else:
        tol = 0.1

    status = "PASS" if max(dQ_diff, dK_diff, dV_diff) < tol else "FAIL"
    print(f"  [{status}] Causal bwd B={B} H={H} R={R} D={D} {dtype}: "
          f"dQ_max={dQ_diff:.6f} dK_max={dK_diff:.6f} dV_max={dV_diff:.6f}")
    assert dQ_diff < tol, f"Causal dQ mismatch! max_diff={dQ_diff} > tol={tol}"
    assert dK_diff < tol, f"Causal dK mismatch! max_diff={dK_diff} > tol={tol}"
    assert dV_diff < tol, f"Causal dV mismatch! max_diff={dV_diff} > tol={tol}"


def test_causal_training_loop():
    """Training loop with is_causal=True: loss should decrease."""
    import metal_flash_sdpa

    B, H, R, D = 1, 4, 64, 64
    target = torch.randn(B, H, R, D, device='mps', dtype=torch.float32)

    proj_in = torch.randn(D, D, device='mps', dtype=torch.float32, requires_grad=True)
    proj_out = torch.randn(D, D, device='mps', dtype=torch.float32, requires_grad=True)

    metal_flash_sdpa.enable()
    losses = []
    lr = 0.001
    for step in range(10):
        x = torch.randn(B, H, R, D, device='mps', dtype=torch.float32)
        q = x @ proj_in
        k = x @ proj_in
        v = x @ proj_in
        out = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)
        out = out @ proj_out
        loss = ((out - target) ** 2).mean()
        loss.backward()
        losses.append(loss.item())
        with torch.no_grad():
            proj_in -= lr * proj_in.grad
            proj_out -= lr * proj_out.grad
            proj_in.grad.zero_()
            proj_out.grad.zero_()
    metal_flash_sdpa.disable()

    print(f"  Causal training losses: {[f'{l:.4f}' for l in losses]}")
    status = "PASS" if losses[-1] < losses[0] else "WARN"
    print(f"  [{status}] Causal training loop: loss {losses[0]:.4f} -> {losses[-1]:.4f}")


def test_causal_dispatch_counter():
    """Verify is_causal=True dispatches to MFA (not falling back)."""
    import metal_flash_sdpa

    metal_flash_sdpa.reset_dispatch_count()
    metal_flash_sdpa.enable()
    Q = torch.randn(1, 4, 512, 64, device='mps', dtype=torch.float16)
    K = torch.randn(1, 4, 512, 64, device='mps', dtype=torch.float16)
    V = torch.randn(1, 4, 512, 64, device='mps', dtype=torch.float16)
    with torch.no_grad():
        for _ in range(3):
            torch.nn.functional.scaled_dot_product_attention(Q, K, V, is_causal=True)
    count = metal_flash_sdpa.get_dispatch_count()
    metal_flash_sdpa.disable()
    assert count == 3, f"Expected 3 causal dispatches, got {count}"
    print(f"  [PASS] Causal dispatch counter: {count} dispatches (expected 3)")


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


def test_backward_timing(B, H, R, C, D, dtype, n_iters=50):
    """Time backward pass: MFA vs default SDPA."""
    import metal_flash_sdpa

    def run_fwd_bwd(Q, K, V):
        out = torch.nn.functional.scaled_dot_product_attention(Q, K, V)
        loss = out.sum()
        loss.backward()

    Q = torch.randn(B, H, R, D, device='mps', dtype=dtype, requires_grad=True)
    K = torch.randn(B, H, C, D, device='mps', dtype=dtype, requires_grad=True)
    V = torch.randn(B, H, C, D, device='mps', dtype=dtype, requires_grad=True)

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

    speedup = default_time / mfa_time if mfa_time > 0 else float('inf')
    print(f"  Fwd+Bwd B={B} H={H} R={R} C={C} D={D} {dtype}: "
          f"default={default_time:.2f}ms  MFA={mfa_time:.2f}ms  speedup={speedup:.2f}x")


if __name__ == '__main__':
    print("=== Metal Flash Attention Tests ===\n")

    print("Forward correctness tests:")
    test_config(1, 1, 64, 64, 64, torch.float16)
    test_config(2, 8, 128, 128, 64, torch.float16)
    test_config(1, 4, 256, 256, 128, torch.float16)
    test_config(4, 8, 512, 512, 64, torch.float16)
    test_config(2, 4, 128, 128, 64, torch.float32)
    test_config(1, 4, 64, 256, 64, torch.float16)
    test_config(1, 4, 256, 64, 64, torch.float16)
    test_config(1, 8, 2048, 2048, 64, torch.float16)

    print("\nBackward correctness tests:")
    test_backward(1, 1, 64, 64, 64, torch.float32)
    test_backward(2, 4, 128, 128, 64, torch.float32)
    test_backward(1, 4, 128, 128, 128, torch.float32)
    test_backward(2, 8, 64, 64, 64, torch.float16)
    test_backward(1, 4, 128, 128, 64, torch.float16)
    test_backward(2, 4, 256, 256, 64, torch.float16)

    print("\nGradient flow test:")
    test_grad_flow()

    print("\nMin sequence length threshold test:")
    test_min_seq_len_threshold()

    print("\nTraining loop test:")
    test_training_loop()

    print("\nBfloat16 forward tests:")
    test_config(1, 4, 256, 256, 64, torch.bfloat16)
    test_config(2, 8, 512, 512, 64, torch.bfloat16)

    print("\nBfloat16 backward tests:")
    test_backward(1, 4, 128, 128, 64, torch.bfloat16)
    test_backward(2, 8, 256, 256, 64, torch.bfloat16)

    print("\nDispatch counter test:")
    import metal_flash_sdpa
    metal_flash_sdpa.reset_dispatch_count()
    metal_flash_sdpa.enable()
    Q = torch.randn(1, 4, 512, 64, device='mps', dtype=torch.float16)
    K = torch.randn(1, 4, 512, 64, device='mps', dtype=torch.float16)
    V = torch.randn(1, 4, 512, 64, device='mps', dtype=torch.float16)
    with torch.no_grad():
        for _ in range(5):
            torch.nn.functional.scaled_dot_product_attention(Q, K, V)
    count = metal_flash_sdpa.get_dispatch_count()
    metal_flash_sdpa.disable()
    assert count == 5, f"Expected 5 dispatches, got {count}"
    print(f"  [PASS] Dispatch counter: {count} dispatches (expected 5)")

    print("\nTrivial mask test:")
    metal_flash_sdpa.reset_dispatch_count()
    metal_flash_sdpa.enable()
    Q = torch.randn(1, 4, 512, 64, device='mps', dtype=torch.float16)
    K = torch.randn(1, 4, 512, 64, device='mps', dtype=torch.float16)
    V = torch.randn(1, 4, 512, 64, device='mps', dtype=torch.float16)
    # All-True boolean mask should be treated as no mask -> MFA dispatches
    all_true_mask = torch.ones(1, 512, dtype=torch.bool, device='mps')
    with torch.no_grad():
        out_masked = torch.nn.functional.scaled_dot_product_attention(Q, K, V, attn_mask=all_true_mask)
        out_none = torch.nn.functional.scaled_dot_product_attention(Q, K, V, attn_mask=None)
    count = metal_flash_sdpa.get_dispatch_count()
    assert count == 2, f"Expected 2 dispatches with all-True mask, got {count}"
    diff = (out_masked - out_none).abs().max().item()
    assert diff < 1e-6, f"All-True mask output differs from None mask: {diff}"
    # Non-trivial mask should fall back
    metal_flash_sdpa.reset_dispatch_count()
    mask_with_false = torch.ones(1, 512, dtype=torch.bool, device='mps')
    mask_with_false[0, 0] = False
    with torch.no_grad():
        torch.nn.functional.scaled_dot_product_attention(Q, K, V, attn_mask=mask_with_false)
    count = metal_flash_sdpa.get_dispatch_count()
    assert count == 0, f"Expected 0 dispatches with non-trivial mask, got {count}"
    metal_flash_sdpa.disable()
    print("  [PASS] Trivial mask: all-True dispatches to MFA, non-trivial falls back")

    print("\nCausal forward correctness tests:")
    test_causal_forward(1, 4, 256, 64, torch.float32)
    test_causal_forward(2, 8, 512, 128, torch.float16)
    test_causal_forward(1, 4, 256, 64, torch.float16)
    test_causal_forward(2, 8, 512, 64, torch.bfloat16)

    print("\nCausal backward correctness tests:")
    test_causal_backward(1, 4, 256, 64, torch.float32)
    test_causal_backward(2, 8, 256, 64, torch.float16)

    print("\nCausal training loop test:")
    test_causal_training_loop()

    print("\nCausal dispatch counter test:")
    test_causal_dispatch_counter()

    print("\nForward timing:")
    test_timing(2, 8, 128, 128, 64, torch.float16)
    test_timing(1, 8, 512, 512, 64, torch.float16)
    test_timing(1, 8, 2048, 2048, 64, torch.float16)

    print("\nBackward timing:")
    test_backward_timing(2, 8, 128, 128, 64, torch.float16)
    test_backward_timing(1, 8, 512, 512, 64, torch.float16)

    print("\n=== All tests passed! ===")
