"""Backward pass correctness tests for Metal Flash Attention."""
import pytest
import torch
import metal_flash_sdpa


def reference_sdpa(Q, K, V, scale):
    """Eager attention on CPU as reference."""
    attn = torch.matmul(Q, K.transpose(-2, -1)) * scale
    attn = torch.softmax(attn, dim=-1)
    return torch.matmul(attn, V)


BACKWARD_CONFIGS = [
    # (B, H, R, C, D, dtype)
    (1, 1, 64, 64, 64, torch.float32),
    (2, 4, 128, 128, 64, torch.float32),
    (1, 4, 128, 128, 128, torch.float32),
    (2, 8, 64, 64, 64, torch.float16),
    (1, 4, 128, 128, 64, torch.float16),
    (2, 4, 256, 256, 64, torch.float16),
    (1, 4, 128, 128, 64, torch.bfloat16),
    (2, 8, 256, 256, 64, torch.bfloat16),
]


def _config_id(config):
    B, H, R, C, D, dtype = config
    dtype_name = {torch.float16: "fp16", torch.float32: "fp32", torch.bfloat16: "bf16"}[dtype]
    return f"{dtype_name}-B{B}-H{H}-{R}x{C}-D{D}"


@pytest.mark.parametrize("B,H,R,C,D,dtype", BACKWARD_CONFIGS, ids=[_config_id(c) for c in BACKWARD_CONFIGS])
def test_backward_correctness(B, H, R, C, D, dtype):
    scale = D ** -0.5

    # CPU reference in float32
    Q_cpu = torch.randn(B, H, R, D, dtype=torch.float32, requires_grad=True)
    K_cpu = torch.randn(B, H, C, D, dtype=torch.float32, requires_grad=True)
    V_cpu = torch.randn(B, H, C, D, dtype=torch.float32, requires_grad=True)

    out_ref = reference_sdpa(Q_cpu, K_cpu, V_cpu, scale)
    grad_out_cpu = torch.randn_like(out_ref)
    out_ref.backward(grad_out_cpu)
    dQ_ref = Q_cpu.grad.clone()
    dK_ref = K_cpu.grad.clone()
    dV_ref = V_cpu.grad.clone()

    # MFA on MPS
    Q_mps = Q_cpu.detach().to(device="mps", dtype=dtype).requires_grad_(True)
    K_mps = K_cpu.detach().to(device="mps", dtype=dtype).requires_grad_(True)
    V_mps = V_cpu.detach().to(device="mps", dtype=dtype).requires_grad_(True)
    grad_out_mps = grad_out_cpu.to(device="mps", dtype=dtype)

    metal_flash_sdpa.enable()
    out_mfa = torch.nn.functional.scaled_dot_product_attention(Q_mps, K_mps, V_mps, scale=scale)
    out_mfa.backward(grad_out_mps)

    dQ_mfa = Q_mps.grad.cpu().float()
    dK_mfa = K_mps.grad.cpu().float()
    dV_mfa = V_mps.grad.cpu().float()

    dQ_diff = (dQ_mfa - dQ_ref).abs().max().item()
    dK_diff = (dK_mfa - dK_ref).abs().max().item()
    dV_diff = (dV_mfa - dV_ref).abs().max().item()

    tol = 1e-3 if dtype == torch.float32 else 0.1
    assert dQ_diff < tol, f"dQ mismatch: max_diff={dQ_diff} > tol={tol}"
    assert dK_diff < tol, f"dK mismatch: max_diff={dK_diff} > tol={tol}"
    assert dV_diff < tol, f"dV mismatch: max_diff={dV_diff} > tol={tol}"
