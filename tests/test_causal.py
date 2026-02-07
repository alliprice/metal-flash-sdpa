"""Causal masking correctness tests for Metal Flash Attention."""
import pytest
import torch
import metal_flash_sdpa


def reference_causal_sdpa(Q, K, V, scale):
    """Eager causal attention on CPU as reference."""
    attn = torch.matmul(Q, K.transpose(-2, -1)) * scale
    R, C = attn.shape[-2], attn.shape[-1]
    causal_mask = torch.triu(torch.ones(R, C, dtype=torch.bool), diagonal=1)
    attn = attn.masked_fill(causal_mask, float("-inf"))
    attn = torch.softmax(attn, dim=-1)
    return torch.matmul(attn, V)


CAUSAL_FORWARD_CONFIGS = [
    # (B, H, R, D, dtype)
    (1, 4, 256, 64, torch.float32),
    (2, 8, 512, 128, torch.float16),
    (1, 4, 256, 64, torch.float16),
    (2, 8, 512, 64, torch.bfloat16),
]


def _fwd_id(config):
    B, H, R, D, dtype = config
    dtype_name = {torch.float16: "fp16", torch.float32: "fp32", torch.bfloat16: "bf16"}[dtype]
    return f"{dtype_name}-B{B}-H{H}-R{R}-D{D}"


@pytest.mark.parametrize("B,H,R,D,dtype", CAUSAL_FORWARD_CONFIGS, ids=[_fwd_id(c) for c in CAUSAL_FORWARD_CONFIGS])
def test_causal_forward(B, H, R, D, dtype):
    Q = torch.randn(B, H, R, D, device="mps", dtype=dtype)
    K = torch.randn(B, H, R, D, device="mps", dtype=dtype)
    V = torch.randn(B, H, R, D, device="mps", dtype=dtype)
    scale = D ** -0.5

    metal_flash_sdpa.enable()
    with torch.no_grad():
        out_mfa = torch.nn.functional.scaled_dot_product_attention(Q, K, V, scale=scale, is_causal=True)

    out_ref = reference_causal_sdpa(
        Q.cpu().float(), K.cpu().float(), V.cpu().float(), scale
    ).to(dtype).to("mps")

    max_diff = (out_mfa - out_ref).abs().max().item()
    tol = 1e-3 if dtype == torch.float32 else 0.05
    assert max_diff < tol, f"Causal forward mismatch: max_diff={max_diff} > tol={tol}"


CAUSAL_BACKWARD_CONFIGS = [
    # (B, H, R, D, dtype)
    (1, 4, 256, 64, torch.float32),
    (2, 8, 256, 64, torch.float16),
]


def _bwd_id(config):
    B, H, R, D, dtype = config
    dtype_name = {torch.float16: "fp16", torch.float32: "fp32", torch.bfloat16: "bf16"}[dtype]
    return f"{dtype_name}-B{B}-H{H}-R{R}-D{D}"


@pytest.mark.parametrize("B,H,R,D,dtype", CAUSAL_BACKWARD_CONFIGS, ids=[_bwd_id(c) for c in CAUSAL_BACKWARD_CONFIGS])
def test_causal_backward(B, H, R, D, dtype):
    scale = D ** -0.5

    Q_cpu = torch.randn(B, H, R, D, dtype=torch.float32, requires_grad=True)
    K_cpu = torch.randn(B, H, R, D, dtype=torch.float32, requires_grad=True)
    V_cpu = torch.randn(B, H, R, D, dtype=torch.float32, requires_grad=True)

    out_ref = reference_causal_sdpa(Q_cpu, K_cpu, V_cpu, scale)
    grad_out_cpu = torch.randn_like(out_ref)
    out_ref.backward(grad_out_cpu)
    dQ_ref = Q_cpu.grad.clone()
    dK_ref = K_cpu.grad.clone()
    dV_ref = V_cpu.grad.clone()

    Q_mps = Q_cpu.detach().to(device="mps", dtype=dtype).requires_grad_(True)
    K_mps = K_cpu.detach().to(device="mps", dtype=dtype).requires_grad_(True)
    V_mps = V_cpu.detach().to(device="mps", dtype=dtype).requires_grad_(True)
    grad_out_mps = grad_out_cpu.to(device="mps", dtype=dtype)

    metal_flash_sdpa.enable()
    out_mfa = torch.nn.functional.scaled_dot_product_attention(Q_mps, K_mps, V_mps, scale=scale, is_causal=True)
    out_mfa.backward(grad_out_mps)

    dQ_mfa = Q_mps.grad.cpu().float()
    dK_mfa = K_mps.grad.cpu().float()
    dV_mfa = V_mps.grad.cpu().float()

    dQ_diff = (dQ_mfa - dQ_ref).abs().max().item()
    dK_diff = (dK_mfa - dK_ref).abs().max().item()
    dV_diff = (dV_mfa - dV_ref).abs().max().item()

    tol = 1e-3 if dtype == torch.float32 else 0.1
    assert dQ_diff < tol, f"Causal dQ mismatch: max_diff={dQ_diff} > tol={tol}"
    assert dK_diff < tol, f"Causal dK mismatch: max_diff={dK_diff} > tol={tol}"
    assert dV_diff < tol, f"Causal dV mismatch: max_diff={dV_diff} > tol={tol}"


def test_causal_dispatch_counter():
    """is_causal=True dispatches to MFA (not falling back)."""
    metal_flash_sdpa.enable()
    Q = torch.randn(1, 4, 512, 64, device="mps", dtype=torch.float16)
    K = torch.randn(1, 4, 512, 64, device="mps", dtype=torch.float16)
    V = torch.randn(1, 4, 512, 64, device="mps", dtype=torch.float16)
    with torch.no_grad():
        for _ in range(3):
            torch.nn.functional.scaled_dot_product_attention(Q, K, V, is_causal=True)
    count = metal_flash_sdpa.get_dispatch_count()
    assert count == 3, f"Expected 3 causal dispatches, got {count}"


def test_causal_training_loop():
    """Causal training loop: loss decreases over 10 steps."""
    B, H, R, D = 1, 4, 64, 64
    target = torch.randn(B, H, R, D, device="mps", dtype=torch.float32)

    proj_in = torch.randn(D, D, device="mps", dtype=torch.float32, requires_grad=True)
    proj_out = torch.randn(D, D, device="mps", dtype=torch.float32, requires_grad=True)

    metal_flash_sdpa.enable()
    losses = []
    lr = 0.001
    for step in range(10):
        x = torch.randn(B, H, R, D, device="mps", dtype=torch.float32)
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

    assert losses[-1] < losses[0], f"Causal loss did not decrease: {losses[0]:.4f} -> {losses[-1]:.4f}"
